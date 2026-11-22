"""Demo web app: password login, then question -> ranked Wikipedia passages.

Environment (set by deploy/demo.py when it creates the pod; never in the image or the repo):
  DEMO_PASSWORD     required; the app refuses to start without it
  SESSION_SECRET    signs the login cookie (random per pod if unset)
  DATA_DIR          demo data folder (default /workspace/demo, mounted read-only)
  AUTO_OFF_HOURS    the pod deletes itself after this many hours (default 3; 0 disables)
  THREADS           CPU threads for torch and faiss (default 8 = the pod's vCPUs)
  RUNPOD_POD_ID, RUNPOD_API_KEY   provided by Runpod to every pod; the key is scoped to this pod only
"""
import hashlib
import hmac
import os
import secrets
import threading
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from search import Searcher

PASSWORD = os.environ.get('DEMO_PASSWORD', '')
if len(PASSWORD) < 8:
    raise SystemExit('DEMO_PASSWORD must be set (8+ characters); refusing to start an open demo')
SECRET = os.environ.get('SESSION_SECRET') or secrets.token_hex(32)
DATA_DIR = os.environ.get('DATA_DIR', '/workspace/demo')
AUTO_OFF_H = float(os.environ.get('AUTO_OFF_HOURS', '3'))
STATIC = Path(__file__).resolve().parent / 'static'
COOKIE = 'demo_session'
RATE = 20                      # searches per minute per client
STARTED = time.time()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)    # no public API explorer
state = {'searcher': None, 'error': None}
hits_by_ip = defaultdict(deque)


def _load():
    try:
        # explicit: inside a Runpod container os.cpu_count() reports the host (e.g. 256), not the pod's vCPUs
        state['searcher'] = Searcher(DATA_DIR, threads=int(os.environ.get('THREADS', '8')))
    except Exception as e:  # noqa: BLE001 - surfaced on /health
        state['error'] = f'{type(e).__name__}: {e}'


def _auto_off():
    """Delete this pod after AUTO_OFF_HOURS, whatever else happens (pod-scoped key, Runpod REST v2)."""
    time.sleep(AUTO_OFF_H * 3600)
    pod, key = os.environ.get('RUNPOD_POD_ID'), os.environ.get('RUNPOD_API_KEY')
    for _ in range(5):
        try:
            req = urllib.request.Request(f'https://api.runpod.io/v2/pods/{pod}', method='DELETE',
                                         headers={'Authorization': f'Bearer {key}'})
            urllib.request.urlopen(req, timeout=20)
        except Exception:  # noqa: BLE001
            time.sleep(30)


@app.on_event('startup')
def startup():
    threading.Thread(target=_load, daemon=True).start()
    if AUTO_OFF_H > 0 and os.environ.get('RUNPOD_POD_ID'):
        threading.Thread(target=_auto_off, daemon=True).start()


def _token():
    day = str(int(time.time() // 86400))
    return day + '.' + hmac.new(SECRET.encode(), day.encode(), hashlib.sha256).hexdigest()


def _logged_in(request):
    tok = request.cookies.get(COOKIE, '')
    day, _, sig = tok.partition('.')
    good = hmac.new(SECRET.encode(), day.encode(), hashlib.sha256).hexdigest()
    return day.isdigit() and abs(int(day) - time.time() // 86400) <= 1 and hmac.compare_digest(sig, good)


def _client(request):
    return request.headers.get('x-forwarded-for', request.client.host if request.client else '?').split(',')[0]


@app.get('/health')
def health():
    s = state['searcher']
    return {'ready': s is not None, 'error': state['error'], 'chunks': len(s) if s else None,
            'load_s': round(s.load_s, 1) if s else None, 'uptime_s': round(time.time() - STARTED),
            'auto_off_hours': AUTO_OFF_H}


@app.get('/')
def index(request: Request):
    return FileResponse(STATIC / ('index.html' if _logged_in(request) else 'login.html'))


@app.post('/login')
def login(request: Request, password: str = Form(...)):
    time.sleep(0.5)                                     # slows down guessing
    if not hmac.compare_digest(password.encode(), PASSWORD.encode()):
        return RedirectResponse('/?bad=1', status_code=303)
    r = RedirectResponse('/', status_code=303)
    r.set_cookie(COOKIE, _token(), httponly=True, secure=os.environ.get('COOKIE_SECURE', '1') == '1',
                 samesite='strict', max_age=86400)        # COOKIE_SECURE=0 only for local http tests
    return r


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    k: int = Field(default=10, ge=1, le=20)


@app.post('/search')
def search(q: Query, request: Request):
    if not _logged_in(request):
        raise HTTPException(401, 'login required')
    s = state['searcher']
    if s is None:
        return JSONResponse({'detail': 'still loading the index, try again in a minute'}, status_code=503)
    win = hits_by_ip[_client(request)]
    now = time.time()
    while win and now - win[0] > 60:
        win.popleft()
    if len(win) >= RATE:
        raise HTTPException(429, 'too many searches, wait a minute')
    win.append(now)
    hits, t = s.search(q.question.strip(), k=q.k)
    return {'question': q.question, 'hits': hits, 'timing_ms': {k: round(v) for k, v in t.items()}}
