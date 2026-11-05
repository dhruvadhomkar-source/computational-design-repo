# Ask the Regs

Ask questions about the UK Building Regulations guidance in plain English and get the
paragraphs that answer them, with the document, paragraph number and printed page for each.

A retrieval-augmented generation (RAG) project by Dhruva Dhomkar, built with Claude Code.

```
$ python -m askregs search "when do flats need more than one staircase" -k 2

1. [AD B Vol 1, para 3.30, p.29] score 0.857
   3.30 Flats should be served by more than one common stair if either of the following applies...
2. [AD B Vol 1, para 3.79, p.38] score 0.822
   3.79 In buildings with a maximum of three storeys above the ground storey, stairs may serve...
```

## Knowledge base

| Document | Chunks |
|---|---|
| Approved Document B, Volume 1: Dwellings (fire safety) | 574 |
| Approved Document M, Volume 1: Dwellings (access) | 157 |
| Approved Document M, Volume 2: Buildings other than dwellings (access) | 265 |

Sources are statutory guidance published on gov.uk; `scripts/download_sources.sh` downloads them.

## How it works

1. **Chunk** (`askregs/ingest.py`): the PDFs are split at numbered paragraphs ("3.30 …"), the unit
   people cite. Running headers and page numbers are removed; each chunk keeps its paragraph number,
   section and printed page label. Paragraphs over 1,500 characters are split into overlapping windows.
2. **Search**, three interchangeable modes (`askregs/search.py`):
   - `bm25`: keyword search, implemented from scratch in `askregs/bm25.py`
   - `dense`: each chunk is embedded as a 384-dimension vector with `BAAI/bge-small-en-v1.5`
     (free, runs on a laptop CPU); chunks are ranked by cosine similarity to the question
   - `hybrid`: both lists merged with Reciprocal Rank Fusion
3. **Evaluate** (`askregs/evaluate.py`): a test set of questions, each with the paragraph that answers it.

## Results

38 questions written in everyday wording (`eval/questions.jsonl`), 996 chunks:

| Mode | Correct paragraph ranked 1st | Correct paragraph in top 5 | MRR | Time per query |
|---|---|---|---|---|
| Keyword (BM25) | 26% | 50% | 0.39 | 1 ms |
| **Embeddings (dense)** | **50%** | **87%** | **0.66** | 65 ms |
| Hybrid (RRF) | 39% | 76% | 0.55 | 56 ms |

**Decision:** dense is the default. Hybrid search is often recommended, but here the keyword
ranking is weak on everyday wording ("staircase" vs the document's "stair") and pulls good dense
results down. Questions that still miss are listed in `eval/results.json` and drive the next changes.

**Caveat:** the test set is small and was written by the developer, so the numbers compare the
modes against each other rather than promising accuracy on real users' questions.

## Run it

```bash
pip install -r requirements.txt
bash scripts/download_sources.sh
python -m askregs ingest                         # PDFs -> data/chunks.jsonl
python -m askregs search "your question" -k 5    # first run embeds all chunks once
python -m askregs.evaluate                       # retrieval metrics per mode
python -m pytest                                 # unit tests
```

## Roadmap

- [x] Paragraph-aware chunking with page-accurate citations
- [x] Keyword, embedding and hybrid retrieval, measured on a test set
- [ ] Answer generation with an LLM: structured JSON output, answers only from retrieved
      paragraphs, every claim cited, "not covered" when the documents don't answer it
- [ ] Answer-quality evaluation (citation accuracy, refusal when unsupported) and cost per question
- [ ] REST API and a React front end
- [ ] Docker image
