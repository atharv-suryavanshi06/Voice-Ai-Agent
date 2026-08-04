flowchart TD

    Start["Call Starts: Agent Greets User"]
    Start --> AskQ["Agent Asks Required Profile Questions (Age, Budget, Diseases, etc.)"]
    AskQ --> UserAnswer["User Speaks Answers (STT: Deepgram)"]
    UserAnswer --> UpdateProfile["Update Profile & Check Missing Fields"]
    UpdateProfile -->|Profile Incomplete| AskQ

    UpdateProfile -->|Profile Complete| MatchEngine["Filter & Score Policies from JSON Catalog"]
    MatchEngine --> RecPolicy["Agent Recommends Best Matching Policies (Gemini + Cartesia)"]

    RecPolicy --> SendMail["Send Recommended Policy Details via Email"]

    RecPolicy --> UserQ["User Asks Detailed Policy Question"]
    UserQ --> STT2["Speech-to-Text (Deepgram)"]
    STT2 --> RAG["Hybrid Retrieval"]
    RAG --> Reranker["Re-Rank Best Document Chunks"]
    Reranker --> Prompt["Inject Policy Context into Prompt"]
    Prompt --> LLM["Gemini Generates Grounded Answer"]
    LLM --> TTS["Cartesia Text-to-Speech"]
    TTS --> Speaker["Speaker Output"]
    Speaker -->|Next Question| UserQ# Voice AI Agent Sequential Flowchart

A simple high-level overview showing the 3 sequential phases of the call:
1. **Information Collection** (Asking profile questions)
2. **Policy Recommendation** (Recommending best fits from JSON catalog)
3. **Deep Query Answering** (Using RAG for follow-up questions)

```mermaid
flowchart TD
    %% Phase 1: Information Collection
    subgraph Phase1 ["Phase 1: Information Collection"]
        Start["📞 Call Starts: Agent Greets User"] --> AskQ["Agent Asks Required Profile Questions<br/>(Age, Budget, Pre-existing Diseases, etc.)"]
        AskQ --> UserAnswer["User Speaks Answers<br/>(STT: Deepgram)"]
        UserAnswer --> UpdateProfile["Update Profile & Check Missing Fields"]
        UpdateProfile -->|Profile Incomplete| AskQ
    end

    %% Phase 2: Recommendation
    subgraph Phase2 ["Phase 2: Policy Recommendation"]
        UpdateProfile -->|Profile Complete| MatchEngine["Filter & Score Policies from JSON Catalog"]
        MatchEngine --> RecPolicy["Agent Recommends Best Matching Policies<br/>(LLM: Gemini | TTS: Cartesia)"]
    end

    %% Phase 3: RAG Follow-up Questions
    subgraph Phase3 ["Phase 3: Policy Q&A (RAG Pipeline)"]
        RecPolicy --> UserQ["User Asks Detailed Policy Question<br/>(e.g., 'What is the waiting period?')"]
        UserQ --> STT2["Speech-to-Text<br/>(Deepgram)"]
        STT2 --> RAG[" Hybrid retrieval"]
        RAG --> Reranker["Re-Rank Best Document Chunks"]
        Reranker --> Prompt["Inject Policy Context into System Prompt"]
        Prompt --> LLM["LLM Answers Grounded in PDF Context<br/>(Gemini)"]
        LLM --> TTS["Text-to-Speech<br/>(Cartesia)"]
        TTS --> Speaker["🔊 Speaker Audio Output"]
        Speaker -->|User Asks Next Question| UserQ
    end

    style Phase1 fill:#e1f5ff,stroke:#01579b,color:#000
    style Phase2 fill:#e8f5e9,stroke:#1b5e20,color:#000
    style Phase3 fill:#fff3e0,stroke:#e65100,color:#000
    style LLM fill:#f3e5f5,stroke:#4a148c,color:#000
    style Speaker fill:#e1f5ff,stroke:#01579b,color:#000
```  