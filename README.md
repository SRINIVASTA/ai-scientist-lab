# 🔬 Autonomous AI Scientist Lab
### Automated Hypothesis Generation, Sandboxed Execution & Research Publishing Framework

This repository serves as a self-contained autonomous machine learning research engine for quantitative financial forecasting, regime-aware feature discovery, and closed-loop scientific publishing.

The architecture is explicitly optimized to ingest historical market OHLCV data via Yahoo Finance (`yfinance`), compute information-theoretic regime features (including rolling **Shannon Entropy**), generate structured hypotheses via a **Groq LLM multi-agent pipeline**, validate candidate models inside an **Abstract Syntax Tree (AST) security sandbox** with self-healing debugging, and automatically open **GitHub Pull Requests** alongside downloadable **PDF/Markdown research papers** and live multi-horizon price forecasts.

### About

**Autonomous AI Scientist Lab** built with **Streamlit**, **Groq Multi-Agent Pipelines (`openai/gpt-oss-120b`)**, and **AST Sandboxed Execution**. Features **live Yahoo Finance market ingestion**, **rolling Shannon Entropy & technical feature engineering**, **self-healing auto-debugging**, **automated GitHub PR & portfolio publishing**, and **live multi-horizon price range forecasting**.

### ⭐ Show Your Support

If you like the UI design, multi-agent research architecture, or quantitative ML engineering behind **Autonomous AI Scientist Lab**, please consider giving this repository a **Star ⭐** on GitHub! It helps showcase this work and motivates future builds.

---
<img src="ai-scientist-lab.png" width="100%">

---

## Architecture Overview

### System Data Flow Architecture
The diagram below illustrates how historical market data and baseline scripts flow through the multi-agent research loop, AST security sandbox, and automated publishing layers:

```mermaid
flowchart TD
    subgraph Data_Ingestion_Layer [Market Data & Baseline Layer]
        A[yfinance API<br>Historical OHLCV Feed] e1@-->|Fetch & Cache| B[(data/market_data.csv<br>Local Market Dataset)]
        C[baselines/baseline_experiment.py<br>Shannon Entropy & ML Baseline] e2@-->|Load Features & Target| D[app.py<br>Central Orchestration Console]
        B e3@--> D
    end

    subgraph Agentic_Research_Loop [Multi-Agent Groq Research Loop]
        D e4@-->|Baseline Code & Memory| E[groq_agents.py<br>Principal Researcher Agent]
        E e5@-->|Structured Hypothesis JSON| F[groq_agents.py<br>ML Engineer Agent]
        F e6@-->|Synthesize Candidate Script| G[sandbox_runner.py<br>AST Security & Execution Sandbox]
        G e7@-->|Runtime Error Traceback| H[groq_agents.py<br>Self-Healing Auto-Debugger]
        H e8@-->|Patched Python Script| G
    end

    subgraph Evaluation_Forecast_Layer [Peer Review & Live Inference Layer]
        G e9@-->|Out-of-Sample Metrics & Plot| I{groq_agents.py<br>Peer Reviewer Critic}
        B e10@-->|Train Multi-Scale RF| J[app.py<br>Live N-Day Market Forecast Card]
    end

    subgraph Publishing_Archive_Layer [Automated Publishing & Artifact Layer]
        I e11@-->|Approved = True| K[groq_agents.py<br>Research Author Agent]
        K e12@-->|Markdown Paper| L[pdf_report.py<br>PDF & Forecast Report Generator]
        K e13@-->|Create Branch & PR| M[github_sync.py<br>GitHub PR & Pages Publisher]
        G e14@-->|Upload Plot & JSON| N[gdrive_store.py<br>Google Drive Artifact Vault]
    end

    %% Running Flow Animation Definitions
    e1@{ animate: true }
    e2@{ animate: true }
    e3@{ animate: true }
    e4@{ animate: true }
    e5@{ animate: true }
    e6@{ animate: true }
    e7@{ animate: true }
    e8@{ animate: true }
    e9@{ animate: true }
    e10@{ animate: true }
    e11@{ animate: true }
    e12@{ animate: true }
    e13@{ animate: true }
    e14@{ animate: true }

    %% Clean Light-Friendly Subgraph Styling (No Black Boxes)
    style Data_Ingestion_Layer fill:#f4f6f9,stroke:#90a4ae,stroke-width:1.5px,color:#1a1a1a
    style Agentic_Research_Loop fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#1a1a1a
    style Evaluation_Forecast_Layer fill:#fff8e1,stroke:#f9a825,stroke-width:1.5px,color:#1a1a1a
    style Publishing_Archive_Layer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a1a1a
```

### Hypothesis Validation & Self-Healing Logic Flow
The Agentic Subsystem validates every candidate script for security, runtime stability, and strict out-of-sample generalization before promoting or publishing changes:

```mermaid
flowchart TD
    Start[Candidate Experiment Script Generated] --> AST_Check{AST Security Scanner<br>Blocked Imports / Calls?}
    
    AST_Check -->|Unsafe Module Detected| Reject_Sec[Block Execution & Log Security Violation]
    AST_Check -->|AST Safe = True| Exec_Subprocess[Execute in Isolated Subprocess]
    
    Exec_Subprocess -->|Runtime / Pandas Error| Debug_Loop{Auto-Debug Retries < 2?}
    Debug_Loop -->|Yes| Fix_Code[Invoke Groq Debugger with Stderr Traceback]
    Fix_Code --> AST_Check
    Debug_Loop -->|No| Log_Fail[Record Failure in Session Experiment Memory]
    
    Exec_Subprocess -->|Execution Success| Critic_Gate{Groq Peer Reviewer<br>val_accuracy > Baseline & No Leakage?}
    Critic_Gate -->|Rejected| Log_Rej[Log Rejection Reason & Iterate Next Attempt]
    Critic_Gate -->|Approved| Publish_Flow[Draft Research Paper, Render PDF & Open GitHub PR]

    %% Clean Light-Friendly Node Styling (No Black Boxes)
    style Start fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#1a1a1a
    style AST_Check fill:#fff8e1,stroke:#f57c00,stroke-width:1.5px,color:#1a1a1a
    style Reject_Sec fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#1a1a1a
    style Debug_Loop fill:#fbe9e7,stroke:#e64a19,stroke-width:1.5px,color:#1a1a1a
    style Critic_Gate fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#1a1a1a
    style Publish_Flow fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a1a1a
```
