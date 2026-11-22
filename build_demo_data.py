"""Build the demo's data folder from the locked night run, then prove the demo search reproduces the evaluation.

Runs once on a CPU pod with the nq-data volume at /workspace. No GPU: no text is embedded here, the existing
contriever vectors are copied row by row.

  1. dedup   keep the first chunk of each distinct text (prefix + body)    -> chunks.parquet, orig_id.npy
  2. vectors copy the matching rows of the night run's embeddings         -> emb.npy (fp16)
  3. bm25    index the kept texts with the evaluation's tokenizer settings -> bm25/
  4. gate D1 run deploy/app/search.py on 500 test questions and compare, question by question, with the locked
             fine-tuned MiniLM result (perq_minilm_ft50.parquet): answer-hit@10 may not drop by more than
             1.5 points and the per-question outcome must agree on >= 93% of questions.

usage: python build_demo_data.py --out /workspace/demo
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXP = Path('/workspace/experiments')
FULL = EXP / 'night_2026-11-21' / 'full'
LOCKED = EXP / 'minilm_2026-11-21' / 'perq_minilm_ft50.parquet'
MODEL = 'contriever-msmarco'
N_GATE, SEED = 500, 13


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == 'darwin' else r / 1e6             # macOS reports bytes, Linux KiB


def dedup(out):
    pf = pq.ParquetFile(FULL / 'chunks.parquet')
    seen, keep_ids, writer, start = set(), [], None, 0
    for rb in pf.iter_batches(batch_size=200_000, columns=['doc_key', 'kind', 'prefix', 'body']):
        b = rb.to_pandas()
        h = pd.util.hash_pandas_object(b.prefix + b.body, index=False).values
        mask = np.zeros(len(b), dtype=bool)
        for i, x in enumerate(h):
            if x not in seen:
                seen.add(x)
                mask[i] = True
        keep_ids.append(start + np.nonzero(mask)[0])
        t = pa.Table.from_pandas(b[mask].reset_index(drop=True), preserve_index=False)
        writer = writer or pq.ParquetWriter(out / 'chunks.parquet', t.schema)
        writer.write_table(t)
        start += len(b)
    writer.close()
    keep = np.concatenate(keep_ids)
    np.save(out / 'orig_id.npy', keep)
    log(f'dedup: {start:,} chunks -> {len(keep):,} distinct ({100 * (1 - len(keep) / start):.1f}% removed)')
    return keep, start


def copy_vectors(out, keep):
    src = np.load(FULL / f'embeddings_{MODEL}.npy', mmap_mode='r')
    dst = np.lib.format.open_memmap(out / 'emb.npy', mode='w+', dtype=np.float16, shape=(len(keep), src.shape[1]))
    for s in range(0, len(keep), 200_000):
        dst[s:s + 200_000] = src[keep[s:s + 200_000]]
    dst.flush()
    log(f'vectors: {dst.shape} fp16 copied from {src.shape}')
    return src.shape[0]


def build_bm25(out):
    import bm25s
    import Stemmer
    t = pq.read_table(out / 'chunks.parquet', columns=['prefix', 'body']).to_pydict()
    texts = [p + b for p, b in zip(t['prefix'], t['body'])]
    del t
    ret = bm25s.BM25()
    ret.index(bm25s.tokenize(texts, stopwords='en', stemmer=Stemmer.Stemmer('english'), show_progress=False),
              show_progress=False)
    ret.save(str(out / 'bm25'))
    log(f'bm25: indexed {len(texts):,} texts; peak RSS {rss_gb():.1f} GB')


def gate(out, threads):
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'app'))
    from search import Searcher
    from textnorm import normalize                      # vendored copy of the metric's normaliser
    s = Searcher(out, threads=threads)
    log(f'searcher loaded in {s.load_s:.0f}s; peak RSS {rss_gb():.1f} GB')
    qs = pd.read_parquet(FULL / 'questions.parquet')
    qs = qs[qs.split == 'test'].sample(N_GATE, random_state=SEED).reset_index(drop=True)
    locked = pd.read_parquet(LOCKED).set_index('qid')
    rows, times = [], []
    for i, q in enumerate(qs.itertuples()):
        hits, t = s.search(q.question, k=10)
        times.append(t)
        bodies = [' ' + normalize(h['text']) + ' ' for h in hits]
        demo = any(f' {a} ' in b for b in bodies for a in q.answers)
        lk = bool(locked.loc[q.qid, 'first_hit'] <= 10)
        rows.append((q.qid, demo, lk))
        if (i + 1) % 100 == 0:
            log(f'  gate {i + 1}/{N_GATE}')
    g = pd.DataFrame(rows, columns=['qid', 'demo_hit10', 'locked_hit10'])
    g.to_parquet(out / 'gate_d1.parquet')
    tm = pd.DataFrame(times)
    rep = {'n': len(g), 'demo_hit@10': 100 * g.demo_hit10.mean(), 'locked_hit@10': 100 * g.locked_hit10.mean(),
           'agreement': 100 * (g.demo_hit10 == g.locked_hit10).mean(),
           'demo_only': int((g.demo_hit10 & ~g.locked_hit10).sum()),
           'locked_only': int((~g.demo_hit10 & g.locked_hit10).sum()),
           'latency_p50_ms': tm.median().round(1).to_dict(), 'latency_p95_ms': tm.quantile(0.95).round(1).to_dict(),
           'load_s': s.load_s, 'peak_rss_gb': rss_gb(), 'threads': threads}
    rep['pass'] = bool(rep['demo_hit@10'] >= rep['locked_hit@10'] - 1.5 and rep['agreement'] >= 93)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--skip-build', action='store_true', help='only re-run gate D1 on an existing folder')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.out / 'meta.json').read_text()) if a.skip_build else {}
    if not a.skip_build:
        t0 = time.time()
        keep, n_src = dedup(a.out)
        copy_vectors(a.out, keep)
        build_bm25(a.out)
        meta = {'source_run': 'night_2026-11-21/full', 'encoder_vectors': f'embeddings_{MODEL}.npy',
                'chunks_source': n_src, 'chunks': int(len(keep)), 'k_rrf': 60, 'w_bm25': 0.5, 'depth': 50,
                'build_s': round(time.time() - t0)}
        (a.out / 'meta.json').write_text(json.dumps(meta, indent=1))
    rep = gate(a.out, a.threads)
    meta['gate_d1'] = rep
    (a.out / 'meta.json').write_text(json.dumps(meta, indent=1))
    log(f"GATE D1 {'PASS' if rep['pass'] else 'FAIL'}: demo hit@10 {rep['demo_hit@10']:.1f} vs locked "
        f"{rep['locked_hit@10']:.1f} on the same {rep['n']} test questions; agreement {rep['agreement']:.1f}%; "
        f"p50 total {rep['latency_p50_ms'].get('total_ms')} ms; peak RSS {rep['peak_rss_gb']:.1f} GB")
    sys.exit(0 if rep['pass'] else 1)


if __name__ == '__main__':
    main()
