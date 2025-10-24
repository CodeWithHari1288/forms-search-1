#!/usr/bin/env python3
"""
retriever.py — LangGraph-friendly Redis retriever (BM25 + Vector + RRF)

Usage from a node:
    from retriever import retrieve
    results = retrieve("address city postcode", top_k=10, form_id="A", mode="rrf")

Env (.env):
    INDEX_NAME=idx:forms
    VECTOR_DIM=1024
    KEY_PREFIX=form:

    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_USERNAME=
    REDIS_PASSWORD=
    REDIS_SSL=false
    REDIS_SSL_CA_CERT=/path/to/ca.pem
    REDIS_SSL_CLIENT_CERT=
    REDIS_SSL_CLIENT_KEY=

    OLLAMA_HOST=http://localhost:11434
    EMBED_MODEL=bge-m3
"""

from __future__ import annotations
import os, ssl, typing as T
import numpy as np
import redis
from dotenv import load_dotenv
from ollama import Client
from redis.commands.search.query import Query

# ---------- ENV ----------
load_dotenv()

INDEX_NAME  = os.getenv("INDEX_NAME", "idx:forms")
VECTOR_DIM  = int(os.getenv("VECTOR_DIM", "1024"))
KEY_PREFIX  = os.getenv("KEY_PREFIX", "form:")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_SSL      = os.getenv("REDIS_SSL", "false").lower() == "true"
SSL_CA   = os.getenv("REDIS_SSL_CA_CERT") or None     # e.g., /home/you/redis-ca.pem
SSL_CERT = os.getenv("REDIS_SSL_CLIENT_CERT") or None # optional (mTLS)
SSL_KEY  = os.getenv("REDIS_SSL_CLIENT_KEY") or None  # optional (mTLS)

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# Which TEXT fields to use for BM25 sparse search.
# Make sure these match your FT schema aliases.
SPARSE_FIELDS = ["text", "question_label", "propertyname_text"]

# ---------- CLIENTS ----------
def _redis_client() -> redis.Redis:
    ssl_ctx = None
    if REDIS_SSL:
        # Build an SSL context; CA may be required to validate server
        ssl_ctx = ssl.create_default_context(cafile=SSL_CA) if SSL_CA else ssl.create_default_context()
        if SSL_CERT and SSL_KEY:
            ssl_ctx.load_cert_chain(certfile=SSL_CERT, keyfile=SSL_KEY)

    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        ssl=REDIS_SSL,
        ssl_cert_reqs=ssl.CERT_REQUIRED if REDIS_SSL else None,
        ssl_ca_certs=SSL_CA if SSL_CA else None,
        ssl_certfile=SSL_CERT or None,
        ssl_keyfile=SSL_KEY or None,
        ssl_context=ssl_ctx if REDIS_SSL else None,
        decode_responses=False,  # keep raw bytes; we normalize later
    )
    r.ping()  # fail fast if not reachable
    return r

def _ollama_client() -> Client:
    c = Client(host=OLLAMA_HOST)
    # Fail fast if model not present
    c.show(EMBED_MODEL)
    return c

# ---------- EMBEDDING HELPERS ----------
def _embed_vec(client: Client, text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Robust embedding extractor (handles nested shapes and key variants)."""
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    vec = resp.get("embedding", resp.get("embeddings"))
    if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
        vec = vec[0]  # flatten [[...]] -> [...]
    if not isinstance(vec, list):
        raise ValueError(f"Unexpected embedding payload type: {type(vec)}")
    if len(vec) != dim:
        raise ValueError(f"Embedding DIM mismatch: expected {dim}, got {len(vec)}")
    # Convert to plain floats (not numpy types) for JSON compatibility
    return [float(x) for x in vec]

def _to_blob(vec: T.List[float]) -> bytes:
    """Convert to float32 bytes for FT.SEARCH KNN."""
    return np.asarray(vec, dtype=np.float32).tobytes()

def _esc_tag(v: str) -> str:
    """Escape spaces for RediSearch TAG filters."""
    return str(v).replace(" ", r"\ ")

# ---------- LOW-LEVEL SEARCH HELPERS ----------
def _bm25_search(
    r: redis.Redis,
    query_text: str,
    top_k: int,
    left_filter: str,
    sparse_fields: list[str],
):
    q = query_text.replace('"', r'\"')
    ors = [f'@{f}:"{q}"' for f in sparse_fields]
    text_part = "(" + " | ".join(ors) + ")" if ors else ""
    expr = (left_filter + " " + text_part).strip() or "*"

    return r.ft(INDEX_NAME).search(
        Query(expr)
        .return_fields(
            "question_label","propertyname","value","pointer",
            "form_id","section","subsection","question_id"
        )
        .paging(0, top_k)
        .dialect(2)
    )

def _knn_search(
    r: redis.Redis,
    blob: bytes,
    top_k: int,
    left_filter: str,
):
    expr = f'{left_filter or "*"} =>[KNN {top_k} @vec $BLOB AS __score]'
    return r.ft(INDEX_NAME).search(
        Query(expr)
        .return_fields(
            "question_label","propertyname","value","pointer",
            "form_id","section","subsection","question_id","__score"
        )
        .sort_by("__score", asc=True)
        .dialect(2)
        .with_params({"BLOB": blob})
    )

def _normalize_doc(d) -> dict:
    def tostr(v):
        return v.decode() if isinstance(v, (bytes, bytearray)) else v
    item = {
        "question_label": tostr(getattr(d, "question_label", "")),
        "propertyname":   tostr(getattr(d, "propertyname", getattr(d, "propertyname_text", ""))),
        "value":          tostr(getattr(d, "value", "")),
        "pointer":        tostr(getattr(d, "pointer", "")),
        "form_id":        tostr(getattr(d, "form_id", "")),
        "section":        tostr(getattr(d, "section", "")),
        "subsection":     tostr(getattr(d, "subsection", "")),
        "question_id":    tostr(getattr(d, "question_id", "")),
    }
    if hasattr(d, "__score"):
        try:
            item["score"] = float(getattr(d, "__score"))
        except Exception:
            pass
    return item

def _rrf_fuse(dense_docs, sparse_docs, k=10, c=60.0) -> list[dict]:
    """Reciprocal Rank Fusion (RRF) — robust hybrid fusion."""
    dlist = [_normalize_doc(d) for d in getattr(dense_docs, "docs", [])]
    slist = [_normalize_doc(d) for d in getattr(sparse_docs, "docs", [])]

    for i, x in enumerate(dlist, start=1): x["_rank_dense"]  = i
    for i, x in enumerate(slist, start=1): x["_rank_sparse"] = i

    def did(x):  # stable doc id for fusion
        return x.get("pointer") or (x.get("form_id","") + "|" + x.get("question_id",""))

    best, score = {}, {}
    for x in dlist:
        id_ = did(x); best.setdefault(id_, x)
        score[id_] = score.get(id_, 0.0) + 1.0/(c + x["_rank_dense"])
    for x in slist:
        id_ = did(x); best.setdefault(id_, x)
        score[id_] = score.get(id_, 0.0) + 1.0/(c + x["_rank_sparse"])

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for id_, sc in ranked:
        item = best[id_].copy()
        item["rrf"] = sc
        out.append(item)
    return out

# ---------- PUBLIC API ----------
def retrieve(
    query_text: str,
    top_k: int = 10,
    *,
    form_id: str | None = None,
    section: str | None = None,
    subsection: str | None = None,
    mode: str = "hybrid",  # "hybrid" | "dense" | "sparse" | "rrf"
    return_fields: T.Iterable[str] = (
        "question_label","propertyname","value","pointer",
        "form_id","section","subsection","question_id","__score"
    ),
) -> T.List[dict]:
    """
    LangGraph-friendly retriever over Redis Stack (RediSearch + RedisJSON + Vector).
    - hybrid: one-call BM25 filter + KNN (vector score sorts)
    - dense:  KNN only
    - sparse: BM25 only
    - rrf:    true two-call hybrid with Reciprocal Rank Fusion
    """
    r = _redis_client()
    ollama = _ollama_client()

    # Build TAG filters
    filters = []
    if form_id:    filters.append(f"@form_id:{{{_esc_tag(form_id)}}}")
    if section:    filters.append(f"@section:{{{_esc_tag(section)}}}")
    if subsection: filters.append(f"@subsection:{{{_esc_tag(subsection)}}}")
    left = " ".join(filters) or "*"

    if mode == "hybrid":
        qvec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(qvec)
        # add sparse clause to restrict candidates
        q = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{q}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr_left = (left + " " + text_part).strip()
        expr = f'{expr_left} =>[KNN {top_k} @vec $BLOB AS __score]'
        res = r.ft(INDEX_NAME).search(
            Query(expr)
            .return_fields(*return_fields)
            .sort_by("__score", asc=True)
            .dialect(2)
            .with_params({"BLOB": blob})
        )
        return [_normalize_doc(d) for d in getattr(res, "docs", [])]

    elif mode == "dense":
        qvec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(qvec)
        res = _knn_search(r, blob, top_k, left)
        return [_normalize_doc(d) for d in getattr(res, "docs", [])]

    elif mode == "sparse":
        res = _bm25_search(r, query_text, top_k, left, SPARSE_FIELDS)
        return [_normalize_doc(d) for d in getattr(res, "docs", [])]

    elif mode == "rrf":
        qvec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(qvec)
        dense_res  = _knn_search(r, blob, top_k, left)
        sparse_res = _bm25_search(r, query_text, top_k, left, SPARSE_FIELDS)
        return _rrf_fuse(dense_res, sparse_res, k=top_k, c=60.0)

    else:
        raise ValueError("mode must be one of: 'hybrid', 'dense', 'sparse', 'rrf'")

# ---------- OPTIONAL: quick local smoke test ----------
if __name__ == "__main__":
    try:
        out = retrieve(
            "address city postcode",
            top_k=5,
            form_id=os.getenv("TEST_FORM_ID", "A"),
            section=os.getenv("TEST_SECTION"),  # e.g., "address"
            mode=os.getenv("TEST_MODE", "rrf"),
        )
        for i, x in enumerate(out, 1):
            print(f"{i:>2}. {x.get('question_label','')} | {x.get('pointer','')} | value={x.get('value','')}")
    except Exception as e:
        print("Error:", e)











def _embed_vec(client: Client, text: str, dim: int | str = VECTOR_DIM) -> list[float]:
    # ensure dim is an int even if env gave us a string
    try:
        dim = int(dim)
    except Exception:
        raise ValueError(f"Bad VECTOR_DIM type: {type(dim)} value={dim!r}")

    resp = client.embeddings(model=EMBED_MODEL, prompt=text)

    # Accept multiple shapes/keys from Ollama
    vec = resp.get("embedding", resp.get("embeddings"))
    if vec is None:
        raise ValueError(f"Embedding payload missing 'embedding'/'embeddings' keys: keys={list(resp.keys())}")

    # Flatten if nested [[...]]
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = vec[0]

    # Convert numpy arrays -> list
    try:
        import numpy as _np
        if isinstance(vec, _np.ndarray):
            vec = vec.tolist()
    except Exception:
        pass

    if not isinstance(vec, list):
        raise ValueError(f"Embedding vector is not a list (got {type(vec)}).")

    n = len(vec)
    if n != dim:
        # Helpful diagnostics if there’s a REAL mismatch
        raise ValueError(
            f"Embedding DIM mismatch: expected {dim} (int), got {n} "
            f"(type(dim)={type(dim)}, nested={isinstance(vec[0], list)})"
        )

    # Ensure plain Python floats for JSON compatibility
    return [float(x) for x in vec]












#!/usr/bin/env python3
"""
retriever.py — Redis (RediSearch + RedisJSON) retriever for LangGraph steps.
Works on redis-py versions that DON'T have Query.with_params / add_param.

Env (.env):
    INDEX_NAME=idx:forms
    VECTOR_DIM=1024
    KEY_PREFIX=form:

    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_USERNAME=
    REDIS_PASSWORD=
    REDIS_SSL=false
    REDIS_SSL_CA_CERT=/path/to/ca.pem
    REDIS_SSL_CLIENT_CERT=
    REDIS_SSL_CLIENT_KEY=

    OLLAMA_HOST=http://localhost:11434
    EMBED_MODEL=bge-m3
"""

from __future__ import annotations
import os, ssl, typing as T
import numpy as np
import redis
from dotenv import load_dotenv
from ollama import Client
from redis.commands.search.query import Query

# ---------------- ENV ----------------
load_dotenv()

INDEX_NAME  = os.getenv("INDEX_NAME", "idx:forms")
VECTOR_DIM  = int(os.getenv("VECTOR_DIM", "1024"))
KEY_PREFIX  = os.getenv("KEY_PREFIX", "form:")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_SSL      = os.getenv("REDIS_SSL", "false").lower() == "true"
SSL_CA   = os.getenv("REDIS_SSL_CA_CERT") or None
SSL_CERT = os.getenv("REDIS_SSL_CLIENT_CERT") or None
SSL_KEY  = os.getenv("REDIS_SSL_CLIENT_KEY") or None

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# BM25 (sparse) fields — aliases must match your FT schema
SPARSE_FIELDS = ["text", "question_label", "propertyname_text"]

# ---------------- Clients ----------------
def _redis_client() -> redis.Redis:
    ssl_ctx = None
    if REDIS_SSL:
        ssl_ctx = ssl.create_default_context(cafile=SSL_CA) if SSL_CA else ssl.create_default_context()
        if SSL_CERT and SSL_KEY:
            ssl_ctx.load_cert_chain(certfile=SSL_CERT, keyfile=SSL_KEY)
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        ssl=REDIS_SSL,
        ssl_cert_reqs=ssl.CERT_REQUIRED if REDIS_SSL else None,
        ssl_ca_certs=SSL_CA if SSL_CA else None,
        ssl_certfile=SSL_CERT or None,
        ssl_keyfile=SSL_KEY or None,
        ssl_context=ssl_ctx if REDIS_SSL else None,
        decode_responses=False,  # keep bytes, we'll decode
    )
    r.ping()
    return r

def _ollama_client() -> Client:
    c = Client(host=OLLAMA_HOST)
    c.show(EMBED_MODEL)  # fail fast if model missing
    return c

# ---------------- Embeddings ----------------
def _embed_vec(client: Client, text: str, dim: int | str = VECTOR_DIM) -> list[float]:
    try:
        dim = int(dim)
    except Exception:
        raise ValueError(f"Bad VECTOR_DIM type: {type(dim)} value={dim!r}")

    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    vec = resp.get("embedding", resp.get("embeddings"))
    if vec is None:
        raise ValueError(f"Embedding payload missing keys; got {list(resp.keys())}")

    # Flatten nested [[...]]
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = vec[0]

    # Numpy → list
    try:
        import numpy as _np
        if isinstance(vec, _np.ndarray):
            vec = vec.tolist()
    except Exception:
        pass

    if not isinstance(vec, list):
        raise ValueError(f"Embedding vector is not a list (got {type(vec)}).")

    n = len(vec)
    if n != dim:
        raise ValueError(f"Embedding DIM mismatch: expected {dim}, got {n}")

    return [float(x) for x in vec]

def _to_blob(vec: T.List[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()

def _esc_tag(v: str) -> str:
    return str(v).replace(" ", r"\ ")

# ---------------- Raw FT.SEARCH helpers (no with_params) ----------------
def _parse_search_result(raw) -> list[dict]:
    """
    Convert raw r.execute_command('FT.SEARCH', ...) response into list[dict].
    """
    if not raw:
        return []
    # raw format: [total, key1, [field, val, field, val, ...], key2, [...], ...]
    out = []
    # raw[0] is total count; we ignore here
    i = 1
    while i < len(raw):
        # key = raw[i]  # bytes key
        fields = raw[i + 1]
        i += 2
        doc = {}
        for j in range(0, len(fields), 2):
            name = fields[j].decode() if isinstance(fields[j], (bytes, bytearray)) else fields[j]
            val  = fields[j + 1]
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = val.decode()
                except Exception:
                    pass
            doc[name] = val
        out.append(doc)
    return out

def _normalize_doc_map(d: dict) -> dict:
    # normalize keys you typically want
    return {
        "question_label": d.get("question_label", ""),
        "propertyname":   d.get("propertyname", d.get("propertyname_text", "")),
        "value":          d.get("value", ""),
        "pointer":        d.get("pointer", ""),
        "form_id":        d.get("form_id", ""),
        "section":        d.get("section", ""),
        "subsection":     d.get("subsection", ""),
        "question_id":    d.get("question_id", ""),
        # include score if present
        "score":          float(d["__score"]) if "__score" in d else None,
        "rrf":            float(d["rrf"]) if "rrf" in d else None,
    }

# Dense (KNN) via raw FT.SEARCH
def _knn_search_raw(r: redis.Redis, expr: str, blob: bytes, return_fields: list[str]):
    return r.execute_command(
        "FT.SEARCH", INDEX_NAME, expr,
        "PARAMS", 2, "BLOB", blob,
        "RETURN", len(return_fields), *return_fields,
        "SORTBY", "__score",
        "DIALECT", 2,
    )

# Sparse (BM25) — we can still use Query here since no PARAMS; or raw as well.
def _bm25_search(r: redis.Redis, expr: str, return_fields: list[str], top_k: int):
    q = (
        Query(expr)
        .return_fields(*[f for f in return_fields if f != "__score"])
        .paging(0, top_k)
        .dialect(2)
    )
    res = r.ft(INDEX_NAME).search(q)
    docs = []
    for d in getattr(res, "docs", []):
        item = {}
        for f in return_fields:
            if f == "__score":  # not returned by sparse
                continue
            v = getattr(d, f, "")
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode()
                except Exception:
                    pass
            item[f] = v
        docs.append(item)
    return docs

def _rrf_fuse(dense_docs: list[dict], sparse_docs: list[dict], k: int = 10, c: float = 60.0) -> list[dict]:
    dlist = dense_docs[:]
    slist = sparse_docs[:]
    for i, x in enumerate(dlist, start=1): x["_rank_dense"]  = i
    for i, x in enumerate(slist, start=1): x["_rank_sparse"] = i

    def did(x: dict) -> str:
        return x.get("pointer") or (x.get("form_id", "") + "|" + x.get("question_id", ""))

    best, score = {}, {}
    for x in dlist:
        id_ = did(x); best.setdefault(id_, x)
        score[id_] = score.get(id_, 0.0) + 1.0 / (c + x["_rank_dense"])
    for x in slist:
        id_ = did(x); best.setdefault(id_, x)
        score[id_] = score.get(id_, 0.0) + 1.0 / (c + x["_rank_sparse"])

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for id_, sc in ranked:
        item = best[id_].copy()
        item["rrf"] = sc
        out.append(item)
    return out

# ---------------- Public API ----------------
def retrieve(
    query_text: str,
    top_k: int = 10,
    *,
    form_id: str | None = None,
    section: str | None = None,
    subsection: str | None = None,
    mode: str = "hybrid",  # "hybrid" | "dense" | "sparse" | "rrf"
    return_fields: T.Iterable[str] = (
        "question_label","propertyname","value","pointer",
        "form_id","section","subsection","question_id","__score"
    ),
) -> T.List[dict]:
    """
    Retrieve results from Redis.
    - hybrid: BM25 clause + KNN in one FT.SEARCH (vector score sorts)
    - dense:  KNN only
    - sparse: BM25 only
    - rrf:    two-call (dense + sparse) then Reciprocal Rank Fusion
    Returns: list of normalized dicts (safe for LangGraph state).
    """
    r = _redis_client()
    ollama = _ollama_client()

    # TAG filters
    filters = []
    if form_id:    filters.append(f"@form_id:{{{_esc_tag(form_id)}}}")
    if section:    filters.append(f"@section:{{{_esc_tag(section)}}}")
    if subsection: filters.append(f"@subsection:{{{_esc_tag(subsection)}}}")
    left = " ".join(filters) or "*"

    fields = list(return_fields)

    if mode == "hybrid":
        # build sparse clause
        q = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{q}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr_left = (left + " " + text_part).strip()

        # embed -> blob and raw FT.SEARCH with PARAMS
        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr = f'{expr_left} =>[KNN {top_k} @vec $BLOB AS __score]'

        raw = _knn_search_raw(r, expr, blob, fields)
        docs = _parse_search_result(raw)
        return [_normalize_doc_map(d) for d in docs]

    elif mode == "dense":
        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr = f'{left or "*"} =>[KNN {top_k} @vec $BLOB AS __score]'
        raw = _knn_search_raw(r, expr, blob, fields)
        docs = _parse_search_result(raw)
        return [_normalize_doc_map(d) for d in docs]

    elif mode == "sparse":
        q = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{q}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr = (left + " " + text_part).strip() or "*"

        docs = _bm25_search(r, expr, fields, top_k)
        return [_normalize_doc_map(d) for d in docs]

    elif mode == "rrf":
        # dense branch
        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr_dense = f'{left or "*"} =>[KNN {top_k} @vec $BLOB AS __score]'
        raw_dense = _knn_search_raw(r, expr_dense, blob, fields)
        dense_docs = [_normalize_doc_map(d) for d in _parse_search_result(raw_dense)]

        # sparse branch
        q = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{q}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr_sparse = (left + " " + text_part).strip() or "*"
        sparse_docs = _bm25_search(r, expr_sparse, fields, top_k)
        sparse_docs = [_normalize_doc_map(d) for d in sparse_docs]

        # fuse
        fused = _rrf_fuse(dense_docs, sparse_docs, k=top_k, c=60.0)
        return fused

    else:
        raise ValueError("mode must be one of: 'hybrid', 'dense', 'sparse', 'rrf'")

# ---------------- Local smoke test ----------------
if __name__ == "__main__":
    try:
        mode = os.getenv("TEST_MODE", "rrf")
        out = retrieve(
            "address city postcode",
            top_k=5,
            form_id=os.getenv("TEST_FORM_ID", "A"),
            section=os.getenv("TEST_SECTION"),  # e.g., "address"
            mode=mode,
        )
        for i, x in enumerate(out, 1):
            print(f"{i:>2}. {x.get('question_label','')} | {x.get('pointer','')} | value={x.get('value','')} | score={x.get('score')} | rrf={x.get('rrf')}")
    except Exception as e:
        print("Error:", e)









#!/usr/bin/env python3
"""
retriever.py — Redis (RediSearch + RedisJSON) retriever for LangGraph steps.
Tolerant to your CSV headers:
    question_label, Prop, Value, Json_pointer, Section, Sub_section

Env (.env):
    INDEX_NAME=idx:forms
    VECTOR_DIM=1024
    KEY_PREFIX=form:
    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_USERNAME=
    REDIS_PASSWORD=
    REDIS_SSL=false
    REDIS_SSL_CA_CERT=/path/to/ca.pem
    REDIS_SSL_CLIENT_CERT=
    REDIS_SSL_CLIENT_KEY=
    OLLAMA_HOST=http://localhost:11434
    EMBED_MODEL=bge-m3
"""

from __future__ import annotations
import os, ssl, typing as T
import numpy as np
import redis
from dotenv import load_dotenv
from ollama import Client
from redis.commands.search.query import Query

# ---------------- ENV ----------------
load_dotenv()

INDEX_NAME  = os.getenv("INDEX_NAME", "idx:forms")
VECTOR_DIM  = int(os.getenv("VECTOR_DIM", "1024"))
KEY_PREFIX  = os.getenv("KEY_PREFIX", "form:")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_SSL      = os.getenv("REDIS_SSL", "false").lower() == "true"
SSL_CA   = os.getenv("REDIS_SSL_CA_CERT") or None
SSL_CERT = os.getenv("REDIS_SSL_CLIENT_CERT") or None
SSL_KEY  = os.getenv("REDIS_SSL_CLIENT_KEY") or None

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# ---------------- Schema/alias tolerance ----------------
# These are the field names we will try (first hit wins) when reading docs back.
FALLBACKS = {
    "question_label": ["question_label", "Question_label", "label"],
    "prop":           ["Prop", "prop", "propertyname", "propertyname_text", "property_name"],
    "value":          ["Value", "value"],
    "pointer":        ["Json_pointer", "json_pointer", "pointer"],
    "section":        ["Section", "section"],
    "subsection":     ["Sub_section", "sub_section", "subsection"],
    "form_id":        ["form_id", "crfId", "crfid"],
    "question_id":    ["question_id", "Question_id"],
    # Redis vector score field
    "score":          ["__score"],
}

# Which TEXT fields to use for BM25 (adjust to your index aliases).
# If you indexed only CSV-exact names, keep "Prop". If you normalized to propertyname_text, add it.
SPARSE_FIELDS = ["text", "question_label", "Prop"]  # add "propertyname_text" if used

# ---------------- Clients ----------------
def _redis_client() -> redis.Redis:
    ssl_ctx = None
    if REDIS_SSL:
        ssl_ctx = ssl.create_default_context(cafile=SSL_CA) if SSL_CA else ssl.create_default_context()
        if SSL_CERT and SSL_KEY:
            ssl_ctx.load_cert_chain(certfile=SSL_CERT, keyfile=SSL_KEY)
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        ssl=REDIS_SSL,
        ssl_cert_reqs=ssl.CERT_REQUIRED if REDIS_SSL else None,
        ssl_ca_certs=SSL_CA if SSL_CA else None,
        ssl_certfile=SSL_CERT or None,
        ssl_keyfile=SSL_KEY or None,
        ssl_context=ssl_ctx if REDIS_SSL else None,
        decode_responses=False,  # keep bytes; we normalize later
    )
    r.ping()
    return r

def _ollama_client() -> Client:
    c = Client(host=OLLAMA_HOST)
    c.show(EMBED_MODEL)  # fail fast if model missing
    return c

# ---------------- Embeddings ----------------
def _embed_vec(client: Client, text: str, dim: int | str = VECTOR_DIM) -> list[float]:
    try:
        dim = int(dim)
    except Exception:
        raise ValueError(f"Bad VECTOR_DIM type: {type(dim)} value={dim!r}")

    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    vec = resp.get("embedding", resp.get("embeddings"))
    if vec is None:
        raise ValueError(f"Embedding payload missing keys; got {list(resp.keys())}")

    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = vec[0]  # flatten [[...]] -> [...]

    try:
        import numpy as _np
        if isinstance(vec, _np.ndarray):
            vec = vec.tolist()
    except Exception:
        pass

    if not isinstance(vec, list):
        raise ValueError(f"Embedding vector is not a list (got {type(vec)}).")

    n = len(vec)
    if n != dim:
        raise ValueError(f"Embedding DIM mismatch: expected {dim}, got {n}")

    return [float(x) for x in vec]

def _to_blob(vec: T.List[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()

def _esc_tag(v: str) -> str:
    return str(v).replace(" ", r"\ ")

# ---------------- FT.SEARCH helpers ----------------
def _parse_search_result(raw) -> list[dict]:
    """
    Convert raw r.execute_command('FT.SEARCH', ...) response into list[dict].
    Format: [total, key1, [field, val, field, val, ...], key2, [...], ...]
    """
    if not raw:
        return []
    out = []
    i = 1
    while i < len(raw):
        # key = raw[i]
        fields = raw[i + 1]
        i += 2
        doc = {}
        for j in range(0, len(fields), 2):
            name = fields[j].decode() if isinstance(fields[j], (bytes, bytearray)) else fields[j]
            val  = fields[j + 1]
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = val.decode()
                except Exception:
                    pass
            doc[name] = val
        out.append(doc)
    return out

def _get_first(d: dict, keys: list[str]) -> str:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode()
                except Exception:
                    pass
            return v
    return ""

def _normalize_doc_map(d: dict) -> dict:
    return {
        "question_label": _get_first(d, FALLBACKS["question_label"]),
        "propertyname":   _get_first(d, FALLBACKS["prop"]),
        "value":          _get_first(d, FALLBACKS["value"]),
        "pointer":        _get_first(d, FALLBACKS["pointer"]),
        "form_id":        _get_first(d, FALLBACKS["form_id"]),
        "section":        _get_first(d, FALLBACKS["section"]),
        "subsection":     _get_first(d, FALLBACKS["subsection"]),
        "question_id":    _get_first(d, FALLBACKS["question_id"]),
        "score":          float(_get_first(d, FALLBACKS["score"])) if _get_first(d, FALLBACKS["score"]) else None,
        "rrf":            float(d["rrf"]) if "rrf" in d else None,
    }

def _knn_search_raw(r: redis.Redis, expr: str, blob: bytes, return_fields: list[str]):
    return r.execute_command(
        "FT.SEARCH", INDEX_NAME, expr,
        "PARAMS", 2, "BLOB", blob,
        "RETURN", len(return_fields), *return_fields,
        "SORTBY", "__score",
        "DIALECT", 2,
    )

def _bm25_search(r: redis.Redis, expr: str, return_fields: list[str], top_k: int):
    q = (
        Query(expr)
        .return_fields(*[f for f in return_fields if f != "__score"])
        .paging(0, top_k)
        .dialect(2)
    )
    res = r.ft(INDEX_NAME).search(q)
    docs = []
    for d in getattr(res, "docs", []):
        row = {}
        for f in return_fields:
            if f == "__score":
                continue
            v = getattr(d, f, "")
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode()
                except Exception:
                    pass
            row[f] = v
        docs.append(row)
    return docs

def _rrf_fuse(dense_docs: list[dict], sparse_docs: list[dict], k: int = 10, c: float = 60.0) -> list[dict]:
    dlist = dense_docs[:]
    slist = sparse_docs[:]
    for i, x in enumerate(dlist, start=1): x["_rank_dense"]  = i
    for i, x in enumerate(slist, start=1): x["_rank_sparse"] = i

    def did(x: dict) -> str:
        return x.get("pointer") or (x.get("form_id", "") + "|" + x.get("question_id", ""))

    best, score = {}, {}
    for x in dlist:
        id_ = did(x); best.setdefault(id_, x)
        score[id_] = score.get(id_, 0.0) + 1.0 / (c + x["_rank_dense"])
    for x in slist:
        id_ = did(x); best.setdefault(id_, x)
        score[id_] = score.get(id_, 0.0) + 1.0 / (c + x["_rank_sparse"])

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for id_, sc in ranked:
        item = best[id_].copy()
        item["rrf"] = sc
        out.append(item)
    return out

# ---------------- Public API ----------------
def retrieve(
    query_text: str,
    top_k: int = 10,
    *,
    form_id: str | None = None,
    section: str | None = None,
    subsection: str | None = None,
    mode: str = "hybrid",  # "hybrid" | "dense" | "sparse" | "rrf"
    # RETURN fields: include your CSV-exact names and common normalized aliases
    return_fields: T.Iterable[str] = (
        "question_label",
        "Prop", "propertyname", "propertyname_text",
        "Value", "value",
        "Json_pointer", "json_pointer", "pointer",
        "Section", "section",
        "Sub_section", "sub_section", "subsection",
        "form_id", "question_id",
        "__score",
    ),
) -> T.List[dict]:
    """
    Retrieve results from Redis with tolerance to your CSV header names.
    """
    r = _redis_client()
    ollama = _ollama_client()

    # TAG filters
    filters = []
    if form_id:    filters.append(f"@form_id:{{{_esc_tag(form_id)}}}")
    if section:    filters.append(f"@section:{{{_esc_tag(section)}}}")
    if subsection: filters.append(f"@subsection:{{{_esc_tag(subsection)}}}")
    left = " ".join(filters) or "*"

    fields = list(return_fields)

    if mode == "hybrid":
        qtxt = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{qtxt}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr_left = (left + " " + text_part).strip()

        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr = f'{expr_left} =>[KNN {top_k} @vec $BLOB AS __score]'

        raw = _knn_search_raw(r, expr, blob, fields)
        docs = _parse_search_result(raw)
        return [_normalize_doc_map(d) for d in docs]

    elif mode == "dense":
        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr = f'{left or "*"} =>[KNN {top_k} @vec $BLOB AS __score]'
        raw = _knn_search_raw(r, expr, blob, fields)
        docs = _parse_search_result(raw)
        return [_normalize_doc_map(d) for d in docs]

    elif mode == "sparse":
        qtxt = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{qtxt}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr = (left + " " + text_part).strip() or "*"
        docs = _bm25_search(r, expr, fields, top_k)
        return [_normalize_doc_map(d) for d in docs]

    elif mode == "rrf":
        # dense
        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr_dense = f'{left or "*"} =>[KNN {top_k} @vec $BLOB AS __score]'
        raw_dense = _knn_search_raw(r, expr_dense, blob, fields)
        dense_docs = [_normalize_doc_map(d) for d in _parse_search_result(raw_dense)]
        # sparse
        qtxt = query_text.replace('"', r'\"')
        ors = [f'@{f}:"{qtxt}"' for f in SPARSE_FIELDS]
        text_part = "(" + " | ".join(ors) + ")" if ors else ""
        expr_sparse = (left + " " + text_part).strip() or "*"
        sparse_docs = _bm25_search(r, expr_sparse, fields, top_k)
        sparse_docs = [_normalize_doc_map(d) for d in sparse_docs]
        # fuse
        return _rrf_fuse(dense_docs, sparse_docs, k=top_k, c=60.0)

    else:
        raise ValueError("mode must be one of: 'hybrid', 'dense', 'sparse', 'rrf'")

# ---------------- Local smoke test ----------------
if __name__ == "__main__":
    try:
        mode = os.getenv("TEST_MODE", "rrf")
        out = retrieve(
            "address city postcode",
            top_k=5,
            form_id=os.getenv("TEST_FORM_ID", "A"),
            section=os.getenv("TEST_SECTION"),  # e.g., "address"
            mode=mode,
        )
        for i, x in enumerate(out, 1):
            print(f"{i:>2}. {x.get('question_label','')} | prop={x.get('propertyname','')} | ptr={x.get('pointer','')} | val={x.get('value','')} | score={x.get('score')} | rrf={x.get('rrf')}")
    except Exception as e:
        print("Error:", e)