graph TD
    subgraph "Ingestion & Indexing Phase"
        PDF[PDF Document] --> Parser[PDF Parser]
        Parser --> RawText[Raw Text]
        RawText --> Chunker[Chunking Strategy]
        
        Chunker --> Chunks[Standard Chunks]
        
        subgraph "Anthropic Contextual Logic"
            Chunks --> ContextLLM[LLM Context Generator]
            RawText --> ContextLLM
            ContextLLM --> ContextualizedChunks[Chunk + Generated Context]
        end
        
        ContextualizedChunks --> EmbedModel[Embedding Model]
        ContextualizedChunks --> BM25[BM25 Encoder]
        
        EmbedModel --> VectorDB[(Vector Store)]
        BM25 --> BM25Index[(BM25 Index)]
    end

    subgraph "Query Phase"
        UserQuery[User Query] --> HybridRetriever[Hybrid Retriever]
        
        HybridRetriever --> VectorSearch[Vector Search]
        HybridRetriever --> KeywordSearch[BM25 Search]
        
        VectorSearch --> TopK1[Results A]
        KeywordSearch --> TopK2[Results B]
        
        TopK1 & TopK2 --> Reranker[Rank Fusion / Dedup]
        Reranker --> FinalContext[Selected Context]
        
        FinalContext --> GenLLM[Generation LLM]
        UserQuery --> GenLLM
        GenLLM --> FinalAnswer[Final Answer + Sources]
    end