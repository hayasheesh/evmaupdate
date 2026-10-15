"""Run the two markets sequentially, with temperature stop/restart and saved progress."""
from datetime import datetime
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTROL = ROOT / 'execute_results/paper_eval_20261014'
TASKKILL = Path(os.environ['SystemRoot']) / 'System32/taskkill.exe'


def temperature():
    p = CONTROL / 'temperature_status.json'
    try:
        x = json.loads(p.read_text())
        x['valid'] = x.get('valid') and time.time() - p.stat().st_mtime < 90
        return x
    except (ValueError, OSError):
        return {'valid': False}


def status(stage, **extra):
    p = HERE / 'status.json'; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(), stage=stage,
                                   supervisor_pid=os.getpid(), **extra), indent=2))
    tmp.replace(p)


def event(name, **extra):
    with (CONTROL / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(), event=name,
                               experiment='ev_realization_count_6station', **extra)) + '\n')


def main():
    if (HERE / 'DONE').exists():
        return
    jobs = [(task, stage) for task in ('aemo', 'ercot') for stage in ('build', 'evaluate')]
    jobs.append(('both', 'report'))
    for task, stage in jobs:
        marker = HERE / f'{task}_{stage}.DONE'
        if marker.exists():
            continue
        while True:
            t = temperature()
            if not (t.get('valid') and t.get('cool_for_restart') and not t.get('hot_reason')) or (CONTROL / 'PAUSE_AUTONOMY').exists() or (HERE / 'PAUSE').exists():
                status('waiting', task=task, next_stage=stage, cpu_c=t.get('cpu_package_c'))
                time.sleep(30)
                continue
            args = [sys.executable, '-X', 'utf8', '-u', str(HERE / 'measure.py'), stage]
            if task != 'both':
                args.append(task)
            log = HERE / f'{task}_{stage}.log'
            with log.open('a', encoding='utf-8') as f:
                child = subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            affinity = ctypes.windll.kernel32.SetProcessAffinityMask
            affinity.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; affinity.restype = ctypes.c_int
            if not affinity(int(child._handle), (1 << 20) | (1 << 21)):
                child.kill()
                raise OSError('cannot set CPU affinity')
            status(stage, task=task, pid=child.pid, log=str(log))
            interrupted = False
            while child.poll() is None:
                t = temperature()
                if not t.get('valid') or t.get('hot_reason') or (CONTROL / 'PAUSE_AUTONOMY').exists() or (HERE / 'PAUSE').exists():
                    subprocess.run([str(TASKKILL), '/PID', str(child.pid), '/T', '/F'], capture_output=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                    interrupted = True
                    event('temperature_pause', task=task, stage=stage, cpu_c=t.get('cpu_package_c'),
                          reason=t.get('hot_reason') or 'sensor/pause')
                    break
                time.sleep(5)
            if interrupted:
                status('cooling', task=task, next_stage=stage)
                time.sleep(30)
                continue
            if child.returncode:
                raise RuntimeError(f'{task} {stage} exited {child.returncode}; see {log}')
            marker.write_text('complete\n')
            if stage == 'evaluate':
                event('experiment_complete', task=task, summary=json.loads((HERE / f'summary_{task}.json').read_text()))
            break
    (HERE / 'DONE').write_text('paired EV 3/4 comparison and plot complete\n')
    status('complete')
    event('experiment_complete', markets=2, sets_per_condition=40,
          figure=str(HERE / 'fig_ev_realization_count.png'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        event('experiment_failed', error=str(exc))
        raise
