"""7 station の作り直しを順に進める（Windows）。

入札（CPU）：市場ごとに学習用25日と検証用5日を同時に作る。同時に作る市場は2つまで。
学習（GPU）：市場の入札ができたら AB を2000回。同時に2本まで。標準 MADDPG（ERCOT）は AB のあと。
CPU は Job Object で抑える（入札 60%、学習 30%、論理32 CPU に対して）。電源設定（boost）には触らない。
状態は status.json、記録は logs/。RUN/STOP を置くと、新しい作業を始めなくなる（動いているものは止めない）。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
WORKER = RUN / 'stage_worker.py'
STATUS = RUN / 'status.json'
LOGS = RUN / 'logs'
PYTHON = sys.executable
BANK_ORDER = ('aemo', 'ercot', 'gb', 'pjm')
TRAIN_ORDER = ('aemo', 'ercot', 'gb', 'pjm', 'ercot_maddpg')
TRAIN_BANK = {'aemo': 'aemo', 'ercot': 'ercot', 'gb': 'gb', 'pjm': 'pjm', 'ercot_maddpg': 'ercot'}
MAX_BANKS = 2
MAX_TRAININGS = 2
BANK_CPU_RATE = 6000      # 1/100 %: 60 %
TRAIN_CPU_RATE = 3000     # 30 %
GPU_FREE_MIB = 5000
CREATE_NO_WINDOW = 0x08000000


class _CpuRate(ctypes.Structure):
    _fields_ = [('ControlFlags', wintypes.DWORD), ('CpuRate', wintypes.DWORD)]


_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
_kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]


def cpu_job(rate: int):
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _CpuRate(0x1 | 0x4, int(rate))      # enable | hard cap
    if not _kernel32.SetInformationJobObject(job, 15, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def gpu_free_mib() -> int:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=60)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


class Orchestrator:
    def __init__(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.bank_job = cpu_job(BANK_CPU_RATE)
        self.train_job = cpu_job(TRAIN_CPU_RATE)
        self.running: dict[str, list[subprocess.Popen]] = {}
        self.state = {
            'project': str(ROOT), 'started_at': now(), 'orchestrator_pid': os.getpid(),
            'power_settings_touched': False, 'banks': {}, 'trainings': {}, 'stage': 'running',
        }
        for task in BANK_ORDER:
            self.state['banks'][task] = {'state': 'pending'}
        for task in TRAIN_ORDER:
            self.state['trainings'][task] = {'state': 'pending'}
        self.save()

    def save(self):
        temp = STATUS.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(STATUS)

    def spawn(self, task: str, stage: str, job) -> subprocess.Popen:
        out = (LOGS / f'{task}_{stage}.stdout.log').open('a', encoding='utf-8')
        err = (LOGS / f'{task}_{stage}.stderr.log').open('a', encoding='utf-8')
        env = os.environ.copy()
        env.update({'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'})
        proc = subprocess.Popen([PYTHON, '-X', 'utf8', '-u', str(WORKER), task, stage], cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                creationflags=CREATE_NO_WINDOW)
        if not _kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            proc.kill()
            raise ctypes.WinError(ctypes.get_last_error())
        return proc

    def start_bank(self, task: str):
        procs = [self.spawn(task, 'train_bank', self.bank_job), self.spawn(task, 'test_bank', self.bank_job)]
        self.running[f'bank:{task}'] = procs
        self.state['banks'][task] = {'state': 'running', 'started_at': now(), 'pids': [p.pid for p in procs]}

    def start_training(self, task: str):
        proc = self.spawn(task, 'pretrain', self.train_job)
        self.running[f'train:{task}'] = [proc]
        self.state['trainings'][task] = {'state': 'running', 'started_at': now(), 'pid': proc.pid}

    def reap(self):
        for key, procs in list(self.running.items()):
            codes = [p.poll() for p in procs]
            if any(code is None for code in codes):
                continue
            kind, task = key.split(':', 1)
            entry = self.state['banks' if kind == 'bank' else 'trainings'][task]
            entry.update({'state': 'done' if all(c == 0 for c in codes) else 'failed',
                          'finished_at': now(), 'exit_codes': codes})
            del self.running[key]

    def step(self) -> bool:
        self.reap()
        stop = (RUN / 'STOP').exists()
        banks_running = sum(1 for k in self.running if k.startswith('bank:'))
        for task in BANK_ORDER:
            if stop or banks_running >= MAX_BANKS:
                break
            if self.state['banks'][task]['state'] == 'pending':
                self.start_bank(task)
                banks_running += 1
        trainings_running = sum(1 for k in self.running if k.startswith('train:'))
        for task in TRAIN_ORDER:
            if stop or trainings_running >= MAX_TRAININGS:
                break
            entry = self.state['trainings'][task]
            bank_state = self.state['banks'][TRAIN_BANK[task]]['state']
            if entry['state'] != 'pending':
                continue
            if bank_state == 'failed':
                entry.update({'state': 'blocked', 'reason': f'bank {TRAIN_BANK[task]} failed'})
                continue
            if bank_state != 'done':
                continue
            if gpu_free_mib() < GPU_FREE_MIB:
                break
            self.start_training(task)
            trainings_running += 1
            time.sleep(120)       # let the first run allocate its replay buffer before the next gate
        self.state['updated_at'] = now()
        self.save()
        return bool(self.running) or (not stop and any(
            e['state'] == 'pending' for e in list(self.state['banks'].values()) + list(self.state['trainings'].values())
        ))

    def run(self):
        try:
            while self.step():
                time.sleep(30)
            self.state['stage'] = 'complete'
        except Exception as exc:
            self.state['stage'] = 'orchestrator_failed'
            self.state['error'] = repr(exc)
            raise
        finally:
            self.state['finished_at'] = now()
            self.save()


if __name__ == '__main__':
    if STATUS.exists() and json.loads(STATUS.read_text(encoding='utf-8')).get('stage') == 'running':
        raise SystemExit('status.json says an orchestrator is already running')
    Orchestrator().run()
