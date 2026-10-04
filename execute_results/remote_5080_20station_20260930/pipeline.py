"""研究室CPUで入札を作り、完了したらRTX 5080でAB学習を起動する。"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
WORKER = RUN / 'stage_worker.py'
STATUS = RUN / 'status.json'


def save(state):
    temp = STATUS.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(STATUS)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock = (RUN / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allowed = sorted(os.sched_getaffinity(0))
    selected = allowed[:max(1,len(allowed)*3//4)]
    os.sched_setaffinity(0,selected)
    state = {'project':str(ROOT),'model':'prod_aemoplan_AB_20station_5080',
             'stations':20,'design_commands':256,'ev_cases':3,'ev_candidates':128,
             'train_days':25,'test_days':5,'target_episode':2000,
             'cpu_affinity':selected,'day_workers':3,'scenario_workers':5,
             'launcher_pid':os.getpid(),'child_pid':None,'stage':'starting',
             'started_at_utc':datetime.now(timezone.utc).isoformat(),
             'completed_at_utc':None,'error':None,'stage_times_seconds':{}}
    save(state)
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'})
    try:
        for stage in ('preflight','train_bank','test_bank','pretrain'):
            state['stage']=stage
            began=time.monotonic()
            with (RUN/f'{stage}.stdout.log').open('a') as out, (RUN/f'{stage}.stderr.log').open('a') as err:
                child=subprocess.Popen([sys.executable,'-u',str(WORKER),stage],cwd=ROOT,env=env,
                                       stdin=subprocess.DEVNULL,stdout=out,stderr=err)
                state['child_pid']=child.pid
                save(state)
                print(f'[pipeline] stage={stage} pid={child.pid}',flush=True)
                code=child.wait()
            state['stage_times_seconds'][stage]=time.monotonic()-began
            if code:
                raise RuntimeError(f'{stage} exited with code {code}')
            save(state)
        state['stage']='complete'
    except Exception as exc:
        state['stage']='failed'
        state['error']=repr(exc)
        raise
    finally:
        state['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
