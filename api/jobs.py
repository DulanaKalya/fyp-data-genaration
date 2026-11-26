"""Background jobs (dataset generation, training, evaluation) started from the UI.
Stored in runs/_ui_jobs.json so they survive server restarts."""
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_FILE = os.path.join(ROOT, "runs", "_ui_jobs.json")


def _load():
    if os.path.exists(JOBS_FILE):
        try:
            return json.load(open(JOBS_FILE))
        except json.JSONDecodeError:
            return []
    return []


def _save(js):
    os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
    tmp = JOBS_FILE + ".tmp"
    json.dump(js, open(tmp, "w"), indent=1)
    os.replace(tmp, JOBS_FILE)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def launch(kind, name, cmd, log_rel, meta=None):
    log_path = os.path.join(ROOT, log_rel)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    js = _load()
    job = {"id": f"{kind}-{int(time.time() * 1000)}", "kind": kind, "name": name, "pid": proc.pid,
           "log": log_rel, "started": time.time(), "meta": meta or {}}
    js.append(job)
    _save(js)
    return job


def tail(log_rel, n=20):
    f = os.path.join(ROOT, log_rel)
    if not os.path.exists(f):
        return ""
    with open(f, errors="replace") as fh:
        return "".join(fh.readlines()[-n:])


def list_jobs(kind=None, with_log=True):
    out = []
    for j in _load():
        if kind and j["kind"] != kind:
            continue
        j = dict(j)
        j["running"] = pid_alive(j["pid"])
        if with_log:
            j["log_tail"] = tail(j["log"], 12)
        out.append(j)
    return out
