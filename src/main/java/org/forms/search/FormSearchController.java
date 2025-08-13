package org.forms.search;

import org.springframework.ai.document.Document;
import org.springframework.web.bind.annotation.*;

import java.util.List;


@RestController
@RequestMapping("/forms")
public class FormSearchController {

        private final ChromaDbService formSearchService;

        public FormSearchController(ChromaDbService formSearchService) {
            this.formSearchService = formSearchService;
        }

        @PostMapping("/add")
        public String addForm(@RequestParam("id") String id, @RequestParam("text") String text) {
            formSearchService.save(id, text);
            return "Form added!";
        }

        @PostMapping("/addBig")
        public String addForm(@RequestBody MultipleFields multipleFields) {
            formSearchService.saveBiggest(multipleFields);
            return "Form added!";
        }

        @GetMapping("/search")
        public List<Document> search(@RequestParam("text") String text) {
            return formSearchService.searchForms(text);
        }
}
