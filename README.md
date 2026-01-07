# Veritas: AI Compliance & Hallucination Auditor 🛡️

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18174089.svg)](https://doi.org/10.5281/zenodo.18174089)

### The Problem: Liability in AI
In high-stakes sectors like Swiss Banking or Healthcare, an AI hallucination isn't just a bug—it's a lawsuit. Standard RAG (Retrieval Augmented Generation) systems often "guess" when they don't know the answer.

### The Solution: Veritas
Veritas is a **Compliance-in-the-Loop** auditing engine. It ingests financial documents (PDFs) and answers queries with strict adherence to source material.

**Key Features:**
- **Zero-Trust Retrieval:** If the data isn't in the document, Veritas returns a generic "FAIL" verdict.
- **Source Citing:** Explicitly references context.
- **Auditable Logs:** Designed for transparent AI governance.

### Technology Stack
- **LangChain:** Orchestration
- **ChromaDB:** Vector Storage
- **OpenAI GPT-4o:** Reasoning Engine
- **Gradio:** User Interface

### How to Run
1. Enter your OpenAI API Key.
2. Upload a PDF (e.g., a 10-K Financial Report).
3. Ask a factual question to see the pass verdict.
4. Ask a trick question to see the hallucination prevention mechanism.
