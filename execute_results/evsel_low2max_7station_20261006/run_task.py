"""1つの task の入札（学習用と検証用を同時に）を作り、両方そろったら学習へ進む（Windows）。

    python run_task.py <task> [train_workers] [test_workers] [scenario_workers]

オーケストレーターの外で1本だけ動かすときに使う。各段は stage_worker.py を子プロセスで起動し、
記録は logs/<task>_<段>.{stdout,stderr}.log に出す。状態は <task>_run_status.json。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
WORKER = RUN / 'stage_worker.py'
CREATE_NO_WINDOW = 0x08000000


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def main() -> int:
    task = sys.argv[1]
    train_workers = sys.argv[2] if len(sys.argv) > 2 else '6'
    test_workers = sys.argv[3] if len(sys.argv) > 3 else '3'
    scenario_workers = sys.argv[4] if len(sys.argv) > 4 else '3'
    status_path = RUN / f'{task}_run_status.json'
    state = {'task': task, 'started_at': now(), 'stages': {}}

    def save():
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def spawn(stage, *extra):
        out = (RUN / 'logs' / f'{task}_{stage}.stdout.log').open('a', encoding='utf-8')
        err = (RUN / 'logs' / f'{task}_{stage}.stderr.log').open('a', encoding='utf-8')
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(WORKER), task, stage, *extra],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                creationflags=CREATE_NO_WINDOW)
        state['stages'][stage] = {'state': 'running', 'pid': proc.pid, 'started_at': now()}
        save()
        return proc

    banks = {'train_bank': spawn('train_bank', train_workers, scenario_workers),
             'test_bank': spawn('test_bank', test_workers, scenario_workers)}
    failed = False
    for stage, proc in banks.items():
        code = proc.wait()
        state['stages'][stage].update(state='done' if code == 0 else 'failed', exit_code=code, finished_at=now())
        failed |= code != 0
        save()
    if failed:
        state['result'] = 'bank failed'
        save()
        return 1
    code = spawn('pretrain').wait()
    state['stages']['pretrain'].update(state='done' if code == 0 else 'failed', exit_code=code, finished_at=now())
    state['result'] = 'complete' if code == 0 else 'pretrain failed'
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
