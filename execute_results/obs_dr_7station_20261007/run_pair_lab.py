"""研究室で回す。入札（学習用と検証用を同時に）を1回作り、そろったら base の学習を起動する。
dr の学習は、GPU の空きが DR_FREE_MIB 以上になったら起動する（20 station の2本と同居するため）。

    python run_pair_lab.py [train_workers] [test_workers] [scenario_workers]

各段は stage_worker.py を子プロセスで起動し、記録は logs/<task>_<段>.{stdout,stderr}.log。状態は run_status.json。
入札は base の設定で作る（dr も同じ入札を使う）。
"""
from __future__ import annotations

from datetime import datetime
import json
import time
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
WORKER = RUN / 'stage_worker_lab.py'
ROOT = Path('/workspace/EVMALOCALUPDATE_obsdr')
DR_FREE_MIB = 3500


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def free_mib() -> int:
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def main() -> int:
    train_workers = sys.argv[1] if len(sys.argv) > 1 else '6'
    test_workers = sys.argv[2] if len(sys.argv) > 2 else '3'
    scenario_workers = sys.argv[3] if len(sys.argv) > 3 else '3'
    status_path = RUN / 'run_status.json'
    state = {'started_at': now(), 'stages': {}}
    (RUN / 'logs').mkdir(exist_ok=True)

    def save():
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def spawn(task, stage, *extra):
        out = (RUN / 'logs' / f'{task}_{stage}.stdout.log').open('a', encoding='utf-8')
        err = (RUN / 'logs' / f'{task}_{stage}.stderr.log').open('a', encoding='utf-8')
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(WORKER), task, stage, *extra],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        state['stages'][f'{task}_{stage}'] = {'state': 'running', 'pid': proc.pid, 'started_at': now()}
        save()
        return proc

    def finish(name, proc):
        code = proc.wait()
        state['stages'][name].update(state='done' if code == 0 else 'failed', exit_code=code, finished_at=now())
        save()
        return code

    for task in ('base', 'dr'):
        if finish(f'{task}_preflight', spawn(task, 'preflight')) != 0:
            state['result'] = f'{task} preflight failed'
            save()
            return 1
    procs = {'base_train_bank': spawn('base', 'train_bank', train_workers, scenario_workers),
             'base_test_bank': spawn('base', 'test_bank', test_workers, scenario_workers)}
    codes = [finish(name, proc) for name, proc in procs.items()]
    if any(codes):
        state['result'] = 'bank failed'
        save()
        return 1
    procs = {'base_pretrain': spawn('base', 'pretrain')}
    time.sleep(900)
    while free_mib() < DR_FREE_MIB and procs['base_pretrain'].poll() is None:
        state['waiting_for_gpu'] = {'free_mib': free_mib(), 'at': now()}
        save()
        time.sleep(600)
    procs['dr_pretrain'] = spawn('dr', 'pretrain')
    codes = [finish(name, proc) for name, proc in procs.items()]
    state['result'] = 'complete' if not any(codes) else 'pretrain failed'
    save()
    return int(any(codes))


if __name__ == '__main__':
    raise SystemExit(main())
