"""End-to-end local test on a 65-chunk synthetic corpus with the real models (no GPU, no volume).

Covers build_demo_data.py (dedup, vector copy, BM25, gate D1) and app/search.py (load + search), then the web app
(login wall, /search). Run: python deploy/tests/test_pipeline_local.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')     # macOS: torch and faiss each ship libomp
if sys.platform == 'darwin':                              # macOS only: faiss + multi-threaded torch segfaults
    os.environ['OMP_NUM_THREADS'] = '1'                   # (two libomp copies); Linux pods are not affected
THREADS = 1 if sys.platform == 'darwin' else 2
import faiss                                              # noqa: E402,F401  must load before torch (see search.py)
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import torch                                              # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'app'))
PROJECT_CLEANING = HERE.parents[1] / 'fresh_start'        # original normaliser, present only in the project repo


def test_textnorm_matches_original():
    from textnorm import normalize
    samples = ['The 1962 -- 1990 era', 'Hôtel <b>California</b>', 'an_apple a day', '≈ ≥', '']
    assert [normalize(s) for s in samples] == ['1962 1990 era', 'hôtel california', 'apple day', '', '']
    if (PROJECT_CLEANING / 'cleaning.py').exists():
        sys.path.insert(0, str(PROJECT_CLEANING))
        import cleaning
        assert all(normalize(s) == cleaning.normalize(s) for s in samples)
    print('textnorm: PASS')


def make_corpus(full):
    bodies = ['Thomas Jefferson drafted the Declaration of Independence in 1776 .', 'Paris is the capital of France .',
              'Thomas Jefferson drafted the Declaration of Independence in 1776 .', 'Berlin is the capital of Germany .',
              'The Eiffel Tower is in Paris .'] + [f'filler text number {i} about nothing' for i in range(60)]
    prefix = ['Declaration | ', 'France | ', 'Declaration | ', 'Germany | ', 'Eiffel | '] + ['Filler | '] * 60
    pd.DataFrame({'doc_key': [f'u{i}' for i in range(len(bodies))], 'kind': 'text', 'prefix': prefix,
                  'body': bodies}).to_parquet(full / 'chunks.parquet')
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    m = AutoModel.from_pretrained('facebook/contriever-msmarco').eval()
    with torch.no_grad():
        enc = tok([p + b for p, b in zip(prefix, bodies)], padding=True, truncation=True, max_length=512,
                  return_tensors='pt')
        h = m(**enc).last_hidden_state
        mk = enc['attention_mask'].unsqueeze(-1).float()
        v = ((h * mk).sum(1) / mk.sum(1)).numpy().astype(np.float16)
    np.save(full / 'embeddings_contriever-msmarco.npy', v)
    pd.DataFrame({'qid': ['q1', 'q2'], 'split': ['test', 'test'],
                  'question': ['who wrote the declaration of independence', 'what is the capital of germany'],
                  'answers': [['thomas jefferson'], ['berlin']], 'doc_key': ['u0', 'u3']}).to_parquet(full / 'questions.parquet')


def main():
    test_textnorm_matches_original()
    tmp = Path(tempfile.mkdtemp())
    full = tmp / 'full'
    full.mkdir()
    make_corpus(full)
    pd.DataFrame({'qid': ['q1', 'q2'], 'first_hit': [1.0, 1.0]}).to_parquet(tmp / 'perq.parquet')

    import build_demo_data as b
    b.FULL, b.LOCKED, b.N_GATE = full, tmp / 'perq.parquet', 2
    sys.argv = ['build', '--out', str(tmp / 'demo'), '--threads', str(THREADS)]
    try:
        b.main()
    except SystemExit as e:
        assert e.code == 0, f'gate D1 failed on the synthetic corpus: exit {e.code}'
    meta = json.loads((tmp / 'demo' / 'meta.json').read_text())
    assert meta['chunks_source'] == 65 and meta['chunks'] == 64, meta        # one exact duplicate removed
    assert np.load(tmp / 'demo' / 'orig_id.npy').tolist() == [0, 1] + list(range(3, 65))
    print('build + gate D1: PASS', meta['gate_d1']['demo_hit@10'], meta['gate_d1']['latency_p50_ms'])

    from search import Searcher
    s = Searcher(tmp / 'demo', threads=THREADS)
    hits, t = s.search('who wrote the declaration of independence', k=3)
    assert hits[0]['path'] == 'Declaration', hits[0]
    assert len({h['chunk'] for h in hits}) == 3                               # no duplicates in results
    print('search: PASS', [(h['path'], round(h['score'], 1)) for h in hits])

    # web app: login wall, wrong password, search
    os.environ.update(DEMO_PASSWORD='correct-horse', DATA_DIR=str(tmp / 'demo'), AUTO_OFF_HOURS='0',
                      COOKIE_SECURE='0', THREADS=str(THREADS))
    from fastapi.testclient import TestClient
    import main as app_main
    with TestClient(app_main.app) as c:
        for _ in range(120):                                                    # index loads in a thread
            if c.get('/health').json()['ready']:
                break
            import time
            time.sleep(0.5)
        assert c.get('/health').json()['ready'], c.get('/health').json()
        assert c.post('/search', json={'question': 'capital of germany'}).status_code == 401
        assert 'login' in c.get('/').text.lower()
        c.post('/login', data={'password': 'wrong-password'})
        assert c.post('/search', json={'question': 'capital of germany'}).status_code == 401
        c.post('/login', data={'password': 'correct-horse'})
        r = c.post('/search', json={'question': 'what is the capital of germany', 'k': 3})
        assert r.status_code == 200, r.text
        assert r.json()['hits'][0]['path'] == 'Germany', r.json()['hits'][0]
        assert c.get('/docs').status_code == 404                                # no public API explorer
    print('web app: PASS')


if __name__ == '__main__':
    main()
