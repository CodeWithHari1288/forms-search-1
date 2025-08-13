package org.forms.search;

import java.util.List;
import java.util.Map;
import org.springframework.ai.chroma.vectorstore.ChromaVectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/search")
public class ChromaDbService {

    private final ChromaVectorStore vectorStore;

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
        doc.getMetadata().put("id", Math.random());
        vectorStore.add(List.of(doc));
    }

    @GetMapping("/query")
    public List<Document> searchForms(@RequestParam String q) {
        SearchRequest request = SearchRequest.builder().query(q).build();

        return vectorStore.similaritySearch(request);
    }
}
