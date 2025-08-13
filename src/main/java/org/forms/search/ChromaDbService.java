package org.forms.search;

import java.nio.charset.StandardCharsets;
import java.util.*;

import org.springframework.ai.chroma.vectorstore.ChromaVectorStore;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.*;

@Service
public class ChromaDbService {

    private final ChromaVectorStore vectorStore;

    @Autowired
    private EmbeddingModel embeddingModel;

    public ChromaDbService(@Autowired ChromaVectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    public void save(String id, String form) {
        String text = form;
        Document doc = new Document(text, Map.of("meta", "metadata"));
        doc.getMetadata().put("id", id);
        vectorStore.add(List.of(doc));
    }

    public void saveBig(MultipleFields fields) {
        MultipleFields text = fields;
        Document doc = new Document(fields.toString(), Map.of("meta", "metadata"));
        doc.getMetadata().put("id", Base64.getEncoder().encode(fields.toString().getBytes(StandardCharsets.UTF_8)));
        vectorStore.add(List.of(doc));
    }

    public void saveBiggest(MultipleFields fields) {
        List<String> chunks = split(fields.toString()); // ~300 tokens per chunk
        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            String form = fields.getFormId();
            System.out.println("111111111 " + chunk);
            String uniqueId = UUID.randomUUID().toString();
            float[] embedding = embeddingModel.embed(chunk);
            Document doc = Document.builder()
                    .id(form + "chunk_" + i)
                    .text(chunk)
                    .metadata(Map.of(
                            "formId", form,
                            "chunkIndex", String.valueOf(i),
                            "completeString", fields.toString()
                    ))
                    .build();
            vectorStore.add(List.of(doc));
        }
    }

    public List<Document> searchForms(@RequestParam String q) {
        SearchRequest request = SearchRequest.builder().query(q)
                .topK(25).similarityThreshold(0.5)
                .build();
        List<Document> results = vectorStore.similaritySearch(request);
        List<Document> addedDocs = new ArrayList<>();
        Set<String> present = new HashSet<>();

        for (Document d : results) {
            String complete = d.getMetadata().get("completeString").toString();
            System.out.println("Full form: " + complete);
            String formId = d.getMetadata().get("formId").toString();

            if(present.contains(formId)){
                continue;
            }
            Document buildDoc = Document.builder()
                    .id(d.getId())
                    .text(complete)
                    .metadata(d.getMetadata())
                    .build();
            addedDocs.add(buildDoc);
            present.add(formId);
        }
        return addedDocs;
    }

    public List<String> split(String text) {
        List<String> tokens = tokenize(text); // custom tokenization
        List<String> chunks = new ArrayList<>();

        int start = 0;
        int chunkSize = 10;
        int overlap = 5;
        while (start < tokens.size()) {
            System.out.println("loop: ");

            int end = Math.min(start + chunkSize, tokens.size());
            chunks.add(String.join(" ", tokens.subList(start, end)));
            start += (chunkSize - overlap);
        }
        System.out.println("Overlap + "+chunks+"222 : "+ tokens);
        return chunks;
    }

    private List<String> tokenize(String text) {
        String[] rawTokens = text.trim().split("-");
        List<String> tokens = new ArrayList<>();
        for (String token : rawTokens) {
            if (!token.isBlank()) tokens.add(token);
        }
        System.out.println("33333333333" +tokens.size());

        return tokens;
    }


    }
