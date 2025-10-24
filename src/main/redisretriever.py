#!/usr/bin/env python3
"""
retriever.py — Redis (RediSearch + RedisJSON) retriever for LangGraph steps.
Fields used (exact, per your CSV & index):
    question_label, Prop, Value, Json_pointer, Section, Sub_section
No section/subsection args — searches across all docs.

Env (.env):
    INDEX_NAME=idx:forms
    VECTOR_DIM=1024
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

# BM25 (sparse) fields — these must be TEXT fields in your FT schema
SPARSE_FIELDS = ["text", "question_label", "Prop"]

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

def _normalize_doc_map(d: dict) -> dict:
    # Only your exact field names
    score = d.get("__score")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    return {
        "question_label": d.get("question_label", ""),
        "Prop":           d.get("Prop", ""),
        "Value":          d.get("Value", ""),
        "Json_pointer":   d.get("Json_pointer", ""),
        "Section":        d.get("Section", ""),
        "Sub_section":    d.get("Sub_section", ""),
        "score":          score,
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
        # Prefer unique pointer if present, else combine a few fields
        return x.get("Json_pointer") or (x.get("Section", "") + "|" + x.get("Sub_section", "") + "|" + x.get("Prop", ""))

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
    mode: str = "hybrid",             # "hybrid" | "dense" | "sparse" | "rrf"
    # RETURN fields: exactly your names + __score
    return_fields: T.Iterable[str] = (
        "question_label", "Prop", "Value", "Json_pointer",
        "Section", "Sub_section",
        "__score",
    ),
) -> T.List[dict]:
    """
    Retrieve results across ALL docs (no section/subsection filters).
    """
    r = _redis_client()
    ollama = _ollama_client()

    # No filters — search across everything
    left = "*"
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
        expr = f'{left} =>[KNN {top_k} @vec $BLOB AS __score]'
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
        # dense branch
        vec = _embed_vec(ollama, query_text, VECTOR_DIM)
        blob = _to_blob(vec)
        expr_dense = f'{left} =>[KNN {top_k} @vec $BLOB AS __score]'
        raw_dense = _knn_search_raw(r, expr_dense, blob, fields)
        dense_docs = [_normalize_doc_map(d) for d in _parse_search_result(raw_dense)]
        # sparse branch
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
            mode=mode,
        )
        for i, x in enumerate(out, 1):
            print(f"{i:>2}. {x.get('question_label','')} | Prop={x.get('Prop','')} | Ptr={x.get('Json_pointer','')} | Val={x.get('Value','')} | score={x.get('score')} | rrf={x.get('rrf')}")
    except Exception as e:
        print("Error:", e)