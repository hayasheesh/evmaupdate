"""GB本線の入札作成・学習・評価を、空き容量と温度を確認して順に実行する。"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONTROL=ROOT/'execute_results'/'paper_eval_20261014'
MODEL='wr_cap77wd_gbplan_6station_resbid_tanh_ls717'
BANK_PREFIX='wr_cap77wd_min70_margin7'
TASKKILL=Path(os.environ['SystemRoot'])/'System32'/'taskkill.exe'


def event(name,**data):
    with (CONTROL/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':name,'experiment':'gb6',**data})+'\n')


def status(stage,**data):
    p=HERE/'status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'time':datetime.now().astimezone().isoformat(),'stage':stage,**data},ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def temperature():
    p=CONTROL/'temperature_status.json'
    try:
        x=json.loads(p.read_text(encoding='utf-8'))
        if time.time()-p.stat().st_mtime>90:x['valid']=False
        return x
    except (OSError,ValueError):return {'valid':False}


def wait_for_capacity(stage,gpu=False):
    while True:
        if (HERE/'PAUSE').exists():
            status('paused',next_stage=stage,reason='GB deferred')
            raise SystemExit(0)
        x=temperature()
        allowed=not (CONTROL/'PAUSE_AUTONOMY').exists() and x.get('valid') and not x.get('hot_reason') and x['cpu_package_c']<80
        if allowed and gpu:
            total,used=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW).strip().split(','))
            allowed=total-used>=5120
        if allowed:return
        status('waiting',next_stage=stage,cpu_package_c=x.get('cpu_package_c'))
        time.sleep(30)


def latest_run():
    runs=sorted((ROOT/'archive').glob(MODEL+'_2*'))
    return runs[-1] if runs else None


def bank_complete(split,count):
    name=f'{BANK_PREFIX}_{split}_{count}_6station_128cmd_3ev_elexon_plan_deviation'
    p=ROOT/'execute_results'/'bid_banks'/name/'manifest.json'
    if not p.exists():return False
    x=json.loads(p.read_text(encoding='utf-8'))
    return x.get('complete') is True and len(x.get('entries',[]))==count


def run_stage(stage,extra=(),training=False):
    wait_for_capacity(stage,gpu=True)
    env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    with (HERE/'logs'/f'{stage}.stdout.log').open('a',encoding='utf-8') as out,(HERE/'logs'/f'{stage}.stderr.log').open('a',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(HERE/'stage_worker.py'),'gb',stage,*extra],
                           cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    status(stage,pid=p.pid)
    event('experiment_stage_started',stage=stage,pid=p.pid)
    requested=False
    while p.poll() is None:
        x=temperature()
        unsafe=x.get('hot_reason') or (not x.get('valid') and x.get('sensor_failures',3)>=3)
        pause=(CONTROL/'PAUSE_AUTONOMY').exists() or (HERE/'PAUSE').exists()
        if (unsafe or pause) and not requested:
            requested=True
            event('temperature_pause' if unsafe else 'experiment_paused',stage=stage,reason=x.get('hot_reason') or 'monitor unavailable / pause',cpu_c=x.get('cpu_package_c'))
            if not training:
                subprocess.run([str(TASKKILL),'/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        if requested and training:
            run=latest_run()
            if run is not None:
                resume=run/'resume';resume.mkdir(exist_ok=True)
                (resume/'STOP_REQUESTED').write_text('temperature/pause stop\n',encoding='utf-8')
        time.sleep(5)
    status(stage,exit_code=p.returncode,paused=requested)
    if p.returncode and not requested:raise RuntimeError(f'{stage} exited with code {p.returncode}')
    return not requested


def main():
    if (HERE/'PAUSE').exists():
        status('paused',reason='GB deferred')
        return
    (HERE/'pipeline.pid').write_text(str(os.getpid()),encoding='ascii')
    for stage,split,count in [('train_bank','train',25),('test_bank','validation',5)]:
        while not bank_complete(split,count):run_stage(stage,('2','2'))
    while True:
        run=latest_run();extra=[]
        if run is not None:
            manifest=run/'resume'/'latest.json'
            if not manifest.exists():raise RuntimeError('existing training has no saved state')
            saved=json.loads(manifest.read_text(encoding='utf-8'))
            if saved['completed_training_episode']>=2000:break
            extra=['--resume-run',str(run)]
        run_stage('pretrain',extra,training=True)
    status('evaluation',run=str(run))
    wait_for_capacity('evaluation')
    out=CONTROL/'main6_gb_v2'
    with (HERE/'logs'/'evaluation.log').open('a',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8','-u',str(CONTROL/'run_eval.py'),'--task','gb',
                        '--stage-dir',str(HERE),'--run-dir',str(run),'--episode','2000',
                        '--output',str(out),'--workers','1','--modes','marl','marl_force'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('complete',run=str(run));event('experiment_complete',run=str(run),stage='evaluation')
    (HERE/'DONE').write_text('complete\n',encoding='utf-8')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));event('coordinator_error',error=str(exc));raise
