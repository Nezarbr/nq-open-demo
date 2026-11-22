# NQ-open retrieval demo

A deployed retrieve-then-rerank system over 2 million Wikipedia passages, answering open-domain questions
from [Natural Questions](https://ai.google.com/research/NaturalQuestions). This repository holds the serving
code: the search pipeline, the web app, the Docker image and its CI/CD, and the on/off control script.

**On held-out questions, an answer appears in the top 10 passages for 91.0% of questions** (95% CI 90.2–91.7,
5,712 test questions), up from 77.2% for the first-stage retriever alone.

## How a question is answered

```
question ─┬─► dense retrieval: contriever-msmarco, exact inner product ──► top 500 ─┐
          └─► BM25 keyword search (English stemming)                    ──► top 500 ─┴─► reciprocal-rank fusion
                                                                                          (k=60, BM25 weight 0.5)
          ─► top 50 ─► cross-encoder reranker (MiniLM, fine-tuned) ─► top 10 passages
```

| Stage | What it adds | answer-hit@10 (test) |
|---|---|---|
| Hybrid retrieval | dense + keyword recall: the answer is in the top 100 for 95.4% of questions | 77.2 |
| Duplicate removal | 38% of passages were exact copies (several revisions of the same article) | 84.2 |
| Reranker, off the shelf | `cross-encoder/ms-marco-MiniLM-L6-v2` | 88.4 |
| **Reranker, fine-tuned** | 3 epochs on 19k training questions, 6 GPU-minutes: [model](https://huggingface.co/NezarIn/nq-open-minilm-reranker) | **91.0** |

*answer-hit@10* is the share of questions where a gold short answer appears in at least one of the top 10
passages. No model in the pipeline has documented Natural Questions training data, apart from the reranker's
fine-tuning on the training split; test questions were never used for any choice.

## Deployment

| Piece | Where | Notes |
|---|---|---|
| Image | `ghcr.io/nezarbr/nq-open-demo` | Built by GitHub Actions on every push to `main`, after a secret scan and an end-to-end test. Models are baked in; no network for models at runtime. |
| Data (6 GB) | Runpod network volume | Passages, fp16 vectors, BM25 index; mounted read-only, never in the image. Built once by `build_demo_data.py`, which also checks that the deployed pipeline reproduces the evaluated one question by question. |
| Compute | Runpod CPU pod, 8 vCPU / 16 GB | Exists only while the demo is on; deleted otherwise. Deletes itself after 3 hours as a fail-safe. |
| Access | Password login | The password is a Runpod secret, injected at start. No public API explorer, per-client rate limit, non-root container. |

The demo is **off by default**. Only the owner can switch it on, from their own machine:

```bash
python demo.py on       # create the pod, wait for /health, print the address (about 3-5 minutes)
python demo.py off      # delete it and verify nothing is left running
```

## Repository

| Path | What |
|---|---|
| `app/search.py` | The pipeline. Every setting matches the evaluated system. |
| `app/main.py` | FastAPI app: login, `/search`, `/health`, auto-off |
| `app/static/` | The two pages (results are rendered as text nodes, never as HTML) |
| `build_demo_data.py` | One-time data build + gate D1 (deployed == evaluated) |
| `demo.py` | On/off control (Runpod REST API; standard library only) |
| `tests/test_pipeline_local.py` | End-to-end test on a tiny synthetic corpus with the real models; runs in CI |
| `Dockerfile`, `.github/workflows/` | Image and CI/CD |

## Limits

- The corpus is the Wikipedia pages behind 100k Natural Questions, not all of Wikipedia: questions on other
  topics will return the nearest available passages.
- answer-hit is a string-match metric, so a passage that mentions the answer by coincidence counts as a hit.
- The demo returns passages; it does not generate an answer sentence.
