"""Temperature-aware reward trials, full learning, then separate final evaluations."""
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


def write_status(stage, **extra):
    p = HERE / 'run_status.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(), stage=stage,
                                   supervisor_pid=os.getpid(), **extra), indent=2))
    temp.replace(p)


def event(name, **extra):
    with (CONTROL / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(), event=name,
                               experiment='afm_full_aemo6', **extra)) + '\n')


def temperature():
    p = CONTROL / 'temperature_status.json'
    try:
        value = json.loads(p.read_text())
        value['valid'] = value.get('valid', False) and time.time() - p.stat().st_mtime < 90
        return value
    except (OSError, ValueError):
        return {'valid': False}


def paused():
    return (HERE / 'PAUSE').exists() or (CONTROL / 'PAUSE_AUTONOMY').exists()


def run_child(args, run, *, stage):
    model = run.name
    while True:
        x = temperature()
        if paused() or not (x.get('valid') and x.get('cool_for_restart') and not x.get('hot_reason')):
            write_status('waiting_for_cooling', model=model, cpu_c=x.get('cpu_package_c'))
            time.sleep(30)
            continue
        used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                          text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip())
        if stage.startswith('training') and used > 12000:
            write_status('waiting_for_gpu', model=model, memory_used_mib=used)
            time.sleep(30)
            continue
        run.mkdir(parents=True, exist_ok=True)
        stop = run / 'STOP_REQUESTED'
        if stop.exists():
            # Retain the previous stop request as a record, without deleting it.
            stop.rename(run / ('STOP_ACK_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
        log = HERE / 'logs' / (model + '_' + stage + '.log')
        log.parent.mkdir(exist_ok=True)
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with log.open('a', encoding='utf-8') as f:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        affinity = ctypes.windll.kernel32.SetProcessAffinityMask
        affinity.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        affinity.restype = ctypes.c_int
        if not affinity(int(child._handle), sum(1 << i for i in range(24, 32))):
            child.kill()
            raise OSError('unable to set process CPU affinity')
        write_status(stage, model=model, pid=child.pid, log=str(log), run=str(run))
        cooling = False
        while child.poll() is None:
            x = temperature()
            if paused() or not x.get('valid') or x.get('hot_reason'):
                if stage.startswith('training'):
                    stop.write_text('temperature/sensor/pause stop\n')
                else:
                    # Evaluation stores each finished pair; restart skips them.
                    child.terminate()
                if not cooling:
                    name = 'temperature_pause' if x.get('hot_reason') else 'algorithm_attention'
                    event(name, model=model, cpu_c=x.get('cpu_package_c'), reason=x.get('hot_reason') or 'sensor/pause')
                cooling = True
            time.sleep(5)
        if cooling:
            write_status('cooling', model=model)
            time.sleep(30)
            continue
        if child.returncode:
            raise RuntimeError(f'{model} {stage} exited {child.returncode}; see {log}')
        return


def train(run, factor, episodes):
    if not (run / 'DONE').exists():
        run_child([sys.executable, '-X', 'utf8', '-u', str(HERE / 'driver.py'), '--run', str(run),
                   '--local-factor', str(factor), '--episodes', str(episodes)], run, stage='training')
    latest = json.loads((run / 'latest.json').read_text())
    if latest['completed_environment_episodes'] != episodes:
        raise ValueError('training completion count mismatch')
    points = [json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
    last = [point for point in points if point['episode'] == episodes]
    if len(last) != 1:
        raise ValueError('validation must identify the actual completed episode')
    return last[0]


def main():
    verification = json.loads((HERE / 'implementation_status.json').read_text())
    if not verification.get('verified'):
        raise ValueError('complete-method verification not passed')
    if (HERE / 'DONE').exists():
        return
    trials = []
    for tag, factor in [('l05', .5), ('l10', 1.), ('l20', 2.)]:
        run = HERE / 'runs' / ('aemo6_full_trial_' + tag)
        point = train(run, factor, 200)
        trials.append(dict(local_factor=factor, run=str(run), **point))
        (HERE / 'reward_trials.json').write_text(json.dumps(trials, indent=2))
    selected = max(trials, key=lambda x: (x['common_bid']['score'], -x['local_factor']))
    (HERE / 'selected_reward.json').write_text(json.dumps(selected, indent=2))
    event('reward_tuning_complete', local_factor=selected['local_factor'], trials=len(trials),
          validation_sum=selected['common_bid']['score'])
    run = HERE / 'runs' / 'aemo6_full_final'
    final = train(run, selected['local_factor'], 10000)
    run_child([sys.executable, '-X', 'utf8', '-u', str(HERE / 'evaluate.py'), '--run', str(run)], run, stage='evaluation')
    points = [json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
    recent = [p['common_bid']['score'] for p in points[-5:]]
    (run / 'convergence_check.json').write_text(json.dumps(dict(last_five_validation_sums=recent,
                                                              range=max(recent) - min(recent),
                                                              interpretation='saved checkpoints; convergence needs assessment'), indent=2))
    (HERE / 'DONE').write_text('full components, reward trials, full training, own/common 40 pairs complete\n')
    write_status('complete', model=run.name, run=str(run), learned_episodes=final['learned_episodes'])
    event('experiment_complete', model=run.name, learned_episodes=final['learned_episodes'],
          sets_per_bid_mode=40)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_status('failed', error=str(exc))
        event('experiment_failed', error=str(exc))
        raise
