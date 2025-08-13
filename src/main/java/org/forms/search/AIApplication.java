package org.forms.search;

import org.springframework.ai.autoconfigure.transformers.TransformersEmbeddingModelAutoConfiguration;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(exclude = {
        TransformersEmbeddingModelAutoConfiguration.class
})
public class AIApplication {
    public static void main(String[] args) {

        SpringApplication.run(AIApplication.class,args);
    }
}