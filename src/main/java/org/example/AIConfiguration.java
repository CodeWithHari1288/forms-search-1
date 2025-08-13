package org.example;

import org.springframework.ai.autoconfigure.vectorstore.chroma.ChromaVectorStoreProperties;
import org.springframework.ai.chroma.vectorstore.ChromaApi;
import org.springframework.ai.chroma.vectorstore.ChromaVectorStore;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AIConfiguration {

    @Bean
    @ConditionalOnMissingBean
    protected ChromaVectorStore vectorStore(
            ChromaVectorStoreProperties properties,
            EmbeddingModel embeddingModel) {
        ChromaApi chromaApi = new ChromaApi("http://localhost:8000");
        return ChromaVectorStore.builder(chromaApi, embeddingModel)
                .initializeSchema(true).build();
    }
}
