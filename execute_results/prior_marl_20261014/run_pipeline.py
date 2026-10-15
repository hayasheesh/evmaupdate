"""検証用データで報酬の3候補を比較し、選んだ条件を2000回学習する。"""
from pathlib import Path
from datetime import datetime
import csv
import ctypes
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONTROL=ROOT/'execute_results/paper_eval_20261014'
TASKKILL=Path(os.environ['SystemRoot'])/'System32/taskkill.exe'
PREFIX='wr_cap77wd_aemoplan_6station_afm_port'


def status(stage,**extra):
    p=HERE/'run_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'time':datetime.now().astimezone().isoformat(),'stage':stage,**extra},ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def event(name,**extra):
    with (CONTROL/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':name,'experiment':'afm_port_aemo6',**extra})+'\n')


def temperature():
    p=CONTROL/'temperature_status.json'
    try:
        x=json.loads(p.read_text(encoding='utf-8'));x['stale']=time.time()-p.stat().st_mtime>=90
        x['valid']=x.get('valid') and not x['stale']
        return x
    except (ValueError,OSError):return {'valid':False}


def latest(model):
    paths=sorted((ROOT/'archive').glob(model+'_2*'))
    return paths[-1] if paths else None


def completed(run):
    p=run/'resume/latest.json' if run else None
    return json.loads(p.read_text(encoding='utf-8'))['completed_training_episode'] if p and p.exists() else 0


def train(model,weight,episodes):
    while True:
        run=latest(model)
        if completed(run)>=episodes:return run
        x=temperature()
        if (HERE/'PAUSE').exists() or (CONTROL/'PAUSE_AUTONOMY').exists() or not (x.get('valid') and not x.get('hot_reason') and x['cpu_package_c']<80):
            status('waiting',model=model,cpu_c=x.get('cpu_package_c'));time.sleep(30);continue
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip())
        if used>12000:status('waiting_for_gpu',memory_used_mib=used);time.sleep(30);continue
        args=[sys.executable,'-X','utf8','-u',str(HERE/'stage_worker.py'),'aemo','pretrain',str(weight),str(episodes),model]
        if run:
            if not (run/'resume/latest.json').exists():raise RuntimeError('previous run lacks a complete resume state')
            args.extend(['--resume-run',str(run)])
        env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
        with (HERE/'logs'/f'{model}.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        affinity=ctypes.windll.kernel32.SetProcessAffinityMask
        affinity.argtypes=[ctypes.c_void_p,ctypes.c_size_t];affinity.restype=ctypes.c_int
        if not affinity(int(child._handle),sum(1<<i for i in range(24,32))):
            child.kill();raise OSError('cannot set training CPU affinity')
        status('training',model=model,local_factor=weight,episodes=episodes,pid=child.pid)
        paused=False
        while child.poll() is None:
            x=temperature()
            if x.get('hot_reason') or x.get('stale') or (not x.get('valid') and x.get('sensor_failures',3)>=3) or (HERE/'PAUSE').exists() or (CONTROL/'PAUSE_AUTONOMY').exists():
                run=latest(model)
                if run:
                    folder=run/'resume';folder.mkdir(exist_ok=True)
                    (folder/'STOP_REQUESTED').write_text('temperature or pause stop\n',encoding='utf-8')
                if not paused:event('temperature_pause',model=model,cpu_c=x.get('cpu_package_c'))
                paused=True
            time.sleep(5)
        run=latest(model)
        if paused:status('cooling',model=model);time.sleep(30);continue
        if child.returncode:raise RuntimeError(f'{model} exited {child.returncode}')
        if completed(run)<episodes:raise RuntimeError('training exited before requested episode count')
        return run


def validation_score(run):
    p=run/'results/test_performance_metrics.csv'
    rows=list(csv.DictReader(p.open(encoding='utf-8')))
    rows=[r for r in rows if int(float(r['Episode']))==200]
    if len(rows)!=1:raise RuntimeError('pilot validation checkpoint200 missing or duplicated')
    r=rows[0]
    return {'global':float(r['Dispatch_Tracking_Rate_%']),'local':float(r['SoC_Hit_Rate_%']),
            'score':float(r['Dispatch_Tracking_Rate_%'])+float(r['SoC_Hit_Rate_%'])}


def main():
    if (HERE/'EXCLUDED_FROM_PAPER.json').exists():
        return
    if (HERE/'DONE').exists():return
    (HERE/'logs').mkdir(exist_ok=True)
    trials=[]
    for tag,weight in [('l05',.5),('l10',1.),('l20',2.)]:
        model=PREFIX+'_tune_'+tag
        run=train(model,weight,200)
        trials.append(dict(validation_score(run),local_factor=weight,model=model,run=str(run)))
        (HERE/'reward_trials.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
    chosen=max(trials,key=lambda x:(x['score'],-x['local_factor']))
    (HERE/'selected_reward.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    event('reward_tuning_complete',trials=trials,selected_factor=chosen['local_factor'])
    tag={.5:'l05',1.:'l10',2.:'l20'}[chosen['local_factor']]
    run=train(PREFIX+'_final_'+tag,chosen['local_factor'],2000)
    status('evaluation',run=str(run))
    env=dict(os.environ,AFM_LOCAL_WEIGHT=str(chosen['local_factor']))
    while True:
        x=temperature()
        if (HERE/'PAUSE').exists() or (CONTROL/'PAUSE_AUTONOMY').exists() or not (x.get('valid') and not x.get('hot_reason') and x['cpu_package_c']<80):
            status('evaluation_waiting');time.sleep(30);continue
        with (HERE/'logs/evaluation.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8',str(CONTROL/'run_eval.py'),'--task','aemo','--stage-dir',str(HERE),
                '--run-dir',str(run),'--episode','2000','--output',str(CONTROL/'prior6_aemo_v2'),
                '--workers','1','--modes','marl','marl_force'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        paused=False
        while child.poll() is None:
            x=temperature()
            if x.get('hot_reason') or x.get('stale') or (not x.get('valid') and x.get('sensor_failures',3)>=3) or (HERE/'PAUSE').exists() or (CONTROL/'PAUSE_AUTONOMY').exists():
                subprocess.run([str(TASKKILL),'/PID',str(child.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                paused=True;status('evaluation_cooling');event('temperature_pause',stage='evaluation',cpu_c=x.get('cpu_package_c'));break
            time.sleep(5)
        if paused:time.sleep(30);continue
        if child.returncode:raise RuntimeError('final evaluation failed')
        break
    (HERE/'DONE').write_text('algorithm port, reward tuning, final2000 and40 pairs complete',encoding='utf-8')
    status('complete',run=str(run),local_factor=chosen['local_factor']);event('experiment_complete',run=str(run),sets=40)


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=str(exc));event('experiment_failed',error=str(exc));raise
