"""PJM の規則の 6 station：入札（学習用と検証用を同時に）を作り、両方そろったら学習へ進む（Windows）。

    python run_task.py [train_workers] [test_workers] [scenario_workers]

入札は優先度を一段下げて作る（同じ計算機の選び方の表の入札を遅らせないため）。学習は通常の優先度。
作り終えた日は飛ばすので、止めても同じ指示で続きから再開できる。
記録は logs/pjm_<段>.{stdout,stderr}.log、状態は pjm_run_status.json。
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
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
NORMAL_PRIORITY_CLASS = 0x00000020


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def main() -> int:
    train_workers = sys.argv[1] if len(sys.argv) > 1 else '4'
    test_workers = sys.argv[2] if len(sys.argv) > 2 else '2'
    scenario_workers = sys.argv[3] if len(sys.argv) > 3 else '3'
    status_path = RUN / 'pjm_run_status.json'
    state = {'task': 'pjm', 'started_at': now(), 'stages': {}}

    def save():
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def spawn(stage, priority, *extra):
        out = (RUN / 'logs' / f'pjm_{stage}.stdout.log').open('a', encoding='utf-8')
        err = (RUN / 'logs' / f'pjm_{stage}.stderr.log').open('a', encoding='utf-8')
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(WORKER), 'pjm', stage, *extra],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                creationflags=CREATE_NO_WINDOW | priority)
        state['stages'][stage] = {'state': 'running', 'pid': proc.pid, 'started_at': now()}
        save()
        return proc

    banks = {'test_bank': spawn('test_bank', BELOW_NORMAL_PRIORITY_CLASS, test_workers, scenario_workers),
             'train_bank': spawn('train_bank', BELOW_NORMAL_PRIORITY_CLASS, train_workers, scenario_workers)}
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
    code = spawn('pretrain', NORMAL_PRIORITY_CLASS).wait()
    state['stages']['pretrain'].update(state='done' if code == 0 else 'failed', exit_code=code, finished_at=now())
    state['result'] = 'complete' if code == 0 else 'pretrain failed'
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
