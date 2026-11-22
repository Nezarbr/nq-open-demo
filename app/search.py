"""Search pipeline of the demo: the same system as the locked evaluation (results/retrieval_2026-11-21).

question -> dense (contriever-msmarco, exact inner product) top 500  \
         -> BM25 (bm25s, English stemmer + stopwords)  top 500        -> RRF (k=60, w_bm25=0.5) -> top DEPTH
         -> fine-tuned MiniLM cross-encoder rerank -> top K

The corpus is deduplicated ahead of time (one row per distinct chunk text), so there is no query-time dedup.
Every setting below matches fresh_start/night/{embed,bakeoff,minilm}.py; change them only with a new evaluation.
"""
import json
import time
from pathlib import Path

import faiss                                 # before torch: on macOS, faiss.add() deadlocks if torch loaded libomp first
import numpy as np
import pyarrow.parquet as pq
import torch

ENCODER = 'facebook/contriever-msmarco'      # mean pooling, no prefix, no normalisation (embed.py MODELS)
RERANKER = 'NezarIn/nq-open-minilm-reranker'
Q_MAXLEN = 64                                # embed.Encoder.queries
RR_MAXLEN = 512                              # minilm.MAXLEN
POOL = 500                                   # per-retriever candidates, as in the evaluation
K_RRF, W_BM25 = 60, 0.5                      # full/hybrid_tuning.json (tuned on dev)
DEPTH = 50                                   # reranking depth chosen on dev


def rrf(dense_ids, bm25_ids, k_rrf=K_RRF, w_bm25=W_BM25, depth=POOL):
    """Reciprocal-rank fusion, identical to bakeoff.rrf for one question (ties keep dense-first order)."""
    s = {}
    for r, c in enumerate(dense_ids):
        s[int(c)] = s.get(int(c), 0.0) + 1.0 / (k_rrf + r + 1)
    for r, c in enumerate(bm25_ids):
        s[int(c)] = s.get(int(c), 0.0) + w_bm25 / (k_rrf + r + 1)
    return sorted(s, key=s.get, reverse=True)[:depth]


class Searcher:
    def __init__(self, data_dir, encoder=ENCODER, reranker=RERANKER, threads=None):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        import bm25s
        import Stemmer
        if threads:
            torch.set_num_threads(threads)
            faiss.omp_set_num_threads(threads)
        d = Path(data_dir)
        t0 = time.time()
        self.meta = json.loads((d / 'meta.json').read_text())
        ch = pq.read_table(d / 'chunks.parquet', columns=['prefix', 'body']).to_pydict()
        self.prefix, self.body = ch['prefix'], ch['body']
        emb = np.load(d / 'emb.npy', mmap_mode='r')
        self.index = faiss.IndexScalarQuantizer(emb.shape[1], faiss.ScalarQuantizer.QT_fp16,
                                                faiss.METRIC_INNER_PRODUCT)   # exact search, vectors kept in fp16
        for s in range(0, emb.shape[0], 500_000):
            self.index.add(np.ascontiguousarray(emb[s:s + 500_000], dtype=np.float32))
        del emb
        self.bm25 = bm25s.BM25.load(str(d / 'bm25'))
        self.stemmer = Stemmer.Stemmer('english')
        self.etok = AutoTokenizer.from_pretrained(encoder)
        self.enc = AutoModel.from_pretrained(encoder).eval()
        self.rtok = AutoTokenizer.from_pretrained(reranker)
        self.rr = AutoModelForSequenceClassification.from_pretrained(reranker, num_labels=1).eval()
        self.load_s = time.time() - t0
        assert self.index.ntotal == len(self.body), (self.index.ntotal, len(self.body))

    def __len__(self):
        return len(self.body)

    def text(self, i):
        return self.prefix[i] + self.body[i]

    @torch.inference_mode()
    def encode(self, questions):
        enc = self.etok(questions, padding=True, truncation=True, max_length=Q_MAXLEN, return_tensors='pt')
        h = self.enc(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy()

    def retrieve(self, question, depth=DEPTH):
        """Hybrid candidates for one question. Returns (chunk ids, timings in ms)."""
        import bm25s
        t = {}
        t0 = time.perf_counter()
        qv = self.encode([question])
        t['encode_ms'] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        _, dids = self.index.search(qv, POOL)
        t['dense_ms'] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        qt = bm25s.tokenize([question], stopwords='en', stemmer=self.stemmer, show_progress=False)
        bids, _ = self.bm25.retrieve(qt, k=min(POOL, len(self)), show_progress=False)
        t['bm25_ms'] = 1000 * (time.perf_counter() - t0)
        return rrf(dids[0][dids[0] >= 0], bids[0])[:depth], t

    @torch.inference_mode()
    def rerank_scores(self, question, ids, batch=64):
        out = []
        for s in range(0, len(ids), batch):
            p = [self.text(i) for i in ids[s:s + batch]]
            enc = self.rtok([question] * len(p), p, truncation='only_second', max_length=RR_MAXLEN,
                            padding=True, return_tensors='pt')
            out.append(self.rr(**enc).logits[:, 0].float().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def search(self, question, k=10, depth=DEPTH):
        ids, t = self.retrieve(question, depth)
        t0 = time.perf_counter()
        sc = self.rerank_scores(question, ids)
        t['rerank_ms'] = 1000 * (time.perf_counter() - t0)
        order = np.argsort(-sc, kind='stable')[:k]
        hits = [{'rank': r + 1, 'chunk': int(ids[j]), 'score': float(sc[j]), 'hybrid_rank': int(j) + 1,
                 'path': self.prefix[ids[j]].rstrip(' |'), 'text': self.body[ids[j]]}
                for r, j in enumerate(order)]
        t['total_ms'] = sum(t.values())
        return hits, t
