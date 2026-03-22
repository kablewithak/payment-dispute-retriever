# Payment Dispute Retriever

Synthetic fintech payment dispute retriever built in phases.

## Case Study

I built this project to learn Weaviate practically rather than passively.

Instead of another toy semantic search demo, I built a fintech payment-dispute retriever with:
- synthetic dispute data
- Weaviate indexing
- BM25, vector, hybrid, and filtered hybrid retrieval
- a FastAPI retrieval API
- an evaluation harness across 50 queries

The key finding was that **hybrid + metadata filters** outperformed a more expensive reranked pipeline on both quality and latency.

Read the full case study here: [docs/CASE_STUDY.md](docs/CASE_STUDY.md)

## Current status

### Phase 1 complete
- synthetic dispute corpus
- eval query seed set
- deterministic generation
- schema validation
- unit tests

### Phase 2 complete
- local Weaviate via Docker
- self-provided vector collection
- Python embedding pipeline
- batch ingestion scripts
- optional live integration test

### Phase 3 complete
- FastAPI retrieval API
- BM25 retrieval
- vector retrieval
- hybrid retrieval
- hybrid retrieval with metadata filters
- structured response formatting
- retrieval unit tests

### Phase 4 complete
- local cross-encoder reranking
- retrieval workflow for reranked search
- eval harness across five retrieval modes
- summary CSV and JSON exports
- confidence calibration fixes

## Setup

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

I built this project to learn Weaviate practically, and the main finding was that metadata-aware hybrid retrieval outperformed a more expensive reranked pipeline on both quality and latency.