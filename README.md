# Payment Dispute Retriever

Synthetic fintech payment dispute retriever built in phases.

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