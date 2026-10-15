"""空いた計算機で、順番待ちの仕事を1つずつ走らせる（2026-10-15〜、Windows と研究室で同じもの）。

    python queue.py <name>          jobs_<name>.json の仕事を上から順に走らせる

仕事の一覧は1つ終わるごとに読み直すので、走らせたまま足せる。済んだ仕事（DONE_<id>）と失敗した仕事（FAILED_<id>）は飛ばす。
失敗しても止めずに次へ進む（見張りが FAILED_ を知らせる）。一覧が尽きたら IDLE_<name> を置いて、10分ごとに読み直す。
各仕事は {"id": 名前, "cwd": ROOT からの場所, "args": [...], "env": {...}, "priority": "below_normal"（Windows だけ）,
"requires": [ROOT からの場所, ...], "min_free_gb": 数, "wait_while_running": [正規表現, ...]}。
requires のファイルがそろい、wait_while_running に当たるプロセスが無く、空きメモリが min_free_gb 以上の状態が
5分おきに3回続くまで（プロセスとメモリは研究室だけ見る）、その仕事の前で待つ（後ろの仕事も待つ）。
空きメモリは始める瞬間だけで見ると、ほかの仕事が一瞬メモリを返した隙に始めてしまうので、続けて満たすことを求める。
args と env の値の {ROOT} はリポジトリの場所に置き換える。実行するのはこのファイルを動かしている python。
記録は logs/<id>.log。
"""
from __future__ import annotations

from datetime import datetime
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_NO_WINDOW = 0x08000000


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def fill(value: str) -> str:
    return str(value).replace('{ROOT}', str(ROOT))


def free_gb() -> float | None:
    """MemAvailable in GB on Linux (the lab); None elsewhere, where min_free_gb is not checked."""

    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    except OSError:
        return None
    return None


def running_matches(patterns) -> list[str]:
    """Command lines of other processes matching any pattern (Linux /proc only)."""

    found = []
    proc = Path('/proc')
    if not patterns or not proc.exists():
        return found
    me = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmd = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
        except OSError:
            continue
        if cmd and 'bash -lc' not in cmd and any(re.search(p, cmd) for p in patterns):
            found.append(cmd[:120])
    return found


def next_job(name: str):
    jobs = json.loads((HERE / f'jobs_{name}.json').read_text(encoding='utf-8'))
    for job in jobs:
        if not ((HERE / f'DONE_{job["id"]}').exists() or (HERE / f'FAILED_{job["id"]}').exists()):
            return job
    return None


def run(job: dict) -> int:
    log = HERE / 'logs' / f'{job["id"]}.log'
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', **{k: fill(v) for k, v in job.get('env', {}).items()})
    flags = 0
    if os.name == 'nt':
        flags = CREATE_NO_WINDOW | (BELOW_NORMAL_PRIORITY_CLASS if job.get('priority') == 'below_normal' else 0)
    with log.open('a', encoding='utf-8') as f:
        f.write(f'{now()} START {job["args"]}\n')
        f.flush()
        code = subprocess.run([sys.executable, '-X', 'utf8', '-u', *[fill(a) for a in job['args']]],
                              cwd=ROOT / job.get('cwd', '.'), env=env, stdout=f, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, creationflags=flags).returncode
        f.write(f'{now()} EXIT {code}\n')
    return code


def main():
    name = sys.argv[1]
    idle = HERE / f'IDLE_{name}'
    ready_checks = 0
    while True:
        job = next_job(name)
        if job is None:
            idle.write_text(now() + '\n', encoding='utf-8')
            time.sleep(600)
            continue
        if idle.exists():
            idle.unlink()
        missing = [r for r in job.get('requires', []) if not (ROOT / fill(r)).exists()]
        missing += [f'running: {c}' for c in running_matches(job.get('wait_while_running', []))]
        free = free_gb()
        if free is not None and free < float(job.get('min_free_gb', 0)):
            missing.append(f'free memory {free:.1f} GB < {job["min_free_gb"]} GB')
        ready_checks = 0 if missing else ready_checks + 1
        if not missing and job.get('min_free_gb') and free is not None and ready_checks < 3:
            missing.append(f'free memory ok {ready_checks}/3')
        if missing:
            (HERE / f'WAITING_{name}').write_text(f'{now()} {job["id"]} {missing}\n', encoding='utf-8')
            time.sleep(300)
            continue
        (HERE / f'WAITING_{name}').unlink(missing_ok=True)
        ready_checks = 0
        (HERE / f'RUNNING_{name}').write_text(f'{now()} {job["id"]}\n', encoding='utf-8')
        code = run(job)
        marker = HERE / (f'DONE_{job["id"]}' if code == 0 else f'FAILED_{job["id"]}')
        marker.write_text(f'{now()} exit {code}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
