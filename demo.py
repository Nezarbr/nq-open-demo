"""Switch the demo on and off from your own machine. Viewers have no switch: only this script (with your key) can.

    python demo.py on       create the pod, wait until the app answers, print its address
    python demo.py status   show whether a demo pod exists and whether it is ready
    python demo.py off      delete the demo pod(s) and verify none is left

Needs (never stored in this repo):
  * Runpod API key: env RUNPOD_API_KEY, or macOS Keychain item 'runpod-api-key'
        security add-generic-password -a "$USER" -s runpod-api-key -w      (prompts for the key)
  * Runpod secret 'nq_demo_password' (Runpod console -> Secrets): the demo password. The pod receives it as
    {{ RUNPOD_SECRET_nq_demo_password }}, so it never passes through this script.
  * demo_config.json next to this file (gitignored; copy demo_config.example.json)

Only pods named exactly NAME are ever touched. Standard library only.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = 'https://api.runpod.io/v2'
NAME = 'nq-demo'
PORT = 8000
HERE = Path(__file__).resolve().parent


def api_key():
    k = os.environ.get('RUNPOD_API_KEY')
    if not k and sys.platform == 'darwin':
        r = subprocess.run(['security', 'find-generic-password', '-a', os.environ.get('USER', ''),
                            '-s', 'runpod-api-key', '-w'], capture_output=True, text=True)
        k = r.stdout.strip() if r.returncode == 0 else None
    if not k:
        sys.exit('no Runpod API key: set RUNPOD_API_KEY or add Keychain item runpod-api-key (see top of demo.py)')
    return k


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={'Authorization': f'Bearer {api_key()}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f'Runpod API {method} {path} -> {e.code}: {e.read().decode()[:500]}')


def config():
    p = HERE / 'demo_config.json'
    if not p.exists():
        sys.exit(f'missing {p.name}: copy demo_config.example.json and fill it in')
    return json.loads(p.read_text())


def demo_pods():
    return [p for p in call('GET', '/pods').get('pods', []) if p.get('name') == NAME]


def url(pod_id):
    return f'https://{pod_id}-{PORT}.proxy.runpod.net'


def health(pod_id):
    try:
        with urllib.request.urlopen(url(pod_id) + '/health', timeout=10) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - not up yet
        return None


def on():
    c = config()
    existing = demo_pods()
    if existing:
        print(f'already on: {url(existing[0]["id"])}  (pod {existing[0]["id"]})')
        return
    body = {'name': NAME, 'image': c['image'], 'cloud': 'SECURE', 'dataCenterIds': [c['data_center']],
            'cpu': {'id': c.get('cpu_flavor', 'cpu3c'), 'vcpuCount': c.get('vcpus', 8)},
            'disk': 20, 'ports': [f'{PORT}/http'],
            'env': {'DEMO_PASSWORD': '{{ RUNPOD_SECRET_nq_demo_password }}', 'DATA_DIR': '/workspace/demo',
                    'THREADS': str(c.get('vcpus', 8)), 'AUTO_OFF_HOURS': str(c.get('auto_off_hours', 3))},
            'mounts': {'network': [{'volumeId': c['volume_id'], 'path': '/workspace'}]}}
    pod = call('POST', '/pods', body)
    pid = pod['id']
    print(f'pod {pid} created ({body["cpu"]["vcpuCount"]} vCPU, ${pod.get("cost", "?")}/h, '
          f'auto-off after {body["env"]["AUTO_OFF_HOURS"]} h); waiting for the app ...')
    t0 = time.time()
    while time.time() - t0 < 1200:
        h = health(pid)
        if h and h.get('ready'):
            print(f'ON after {time.time() - t0:.0f}s: {url(pid)}  ({h["chunks"]:,} passages, index loaded in {h["load_s"]}s)')
            return
        if h and h.get('error'):
            sys.exit(f'app failed to load: {h["error"]} - run "python demo.py off"')
        time.sleep(15)
        print(f'  ... {time.time() - t0:.0f}s ({"loading index" if h else "starting container"})', flush=True)
    sys.exit(f'not ready after 20 min; pod {pid} kept for inspection - run "python demo.py off" when done')


def status():
    pods = demo_pods()
    if not pods:
        print('OFF (no demo pod)')
    for p in pods:
        h = health(p['id'])
        print(f'pod {p["id"]}: {p.get("desiredStatus") or p.get("status")}, app '
              f'{"ready" if h and h.get("ready") else "not ready"}: {url(p["id"])}')


def off():
    pods = demo_pods()
    for p in pods:
        call('DELETE', f'/pods/{p["id"]}')
        print(f'deleted pod {p["id"]}')
    for _ in range(12):
        if not demo_pods():
            print('OFF: no demo pod left')
            return
        time.sleep(5)
    sys.exit('a demo pod is still listed - check the Runpod console')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    {'on': on, 'off': off, 'status': status}.get(cmd, lambda: sys.exit(__doc__))()
