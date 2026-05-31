# Multi-Agent Underwriting System

A LangGraph-based Multi-Agent System (MAS) that decomposes insurance underwriting into specialized sub-agents — each responsible for a distinct domain of risk assessment. Agents communicate, self-correct, and synthesize findings into a structured underwriting report.

## Agent architecture

```
                    ┌─────────────────────┐
                    │  Orchestrator Agent  │
                    │  (LangGraph Router)  │
                    └──────────┬──────────┘
           ┌──────────┬────────┴────────┬──────────┐
           ▼          ▼                 ▼          ▼
    ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
    │  Research  │ │    Risk    │ │Compliance│ │    Report    │
    │   Agent    │ │Assessment  │ │  Agent   │ │  Generator   │
    │            │ │   Agent    │ │          │ │    Agent     │
    └────────────┘ └────────────┘ └──────────┘ └──────────────┘
           │              │              │              │
           └──────────────┴──────────────┴──────────────┘
                              ▼
                    Structured Underwriting Report
```

## Features

- **LangGraph state machine** — typed state passed between agents, full graph visualization
- **Specialized sub-agents** — research, risk scoring, compliance check, report generation
- **Self-reflection loops** — agents can flag uncertainty and trigger re-evaluation
- **Function Calling** — agents query SQL databases and external APIs via tool use
- **Structured output** — Pydantic-validated JSON underwriting reports

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
python run_underwriting.py --applicant sample_data/applicant_001.json
```

## Stack

`LangGraph` `LangChain` `OpenAI / Azure OpenAI` `Pydantic` `Python 3.11`
