graph LR
    %% Styles mimicking a clean UI %%
    classDef trigger fill:#2ecc71,stroke:#27ae60,color:white,rx:5,ry:5;
    classDef router fill:#f39c12,stroke:#d35400,color:white,rx:5,ry:5;
    classDef model fill:#3498db,stroke:#2980b9,color:white,rx:5,ry:5;
    classDef db fill:#9b59b6,stroke:#8e44ad,color:white,rx:5,ry:5;
    classDef output fill:#34495e,stroke:#2c3e50,color:white,rx:5,ry:5;

    %% Nodes %%
    Start((User Query)):::trigger
    LangDetect[⚙️ Language Detector]:::router
    
    %% The Branching (Switch) %%
    BERT_EN[🇺🇸 BERT English]:::model
    BERT_ES[🇪🇸 BERT Spanish]:::model
    BERT_Multi[🌐 Multilingual BERT]:::model

    %% The Convergence (RAG) %%
    VectorDB[(🗄️ Vector Store)]:::db
    Retriever[🔍 RAG Retriever]:::model
    LLM[🤖 Synthesis LLM]:::output
    
    %% Connections %%
    Start --> LangDetect
    LangDetect -- "English" --> BERT_EN
    LangDetect -- "Spanish" --> BERT_ES
    LangDetect -- "Other" --> BERT_Multi

    %% Logic Flow %%
    BERT_EN --> Retriever
    BERT_ES --> Retriever
    BERT_Multi --> Retriever

    VectorDB -.- Retriever
    Retriever --> LLM