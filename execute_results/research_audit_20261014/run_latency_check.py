"""同じ制御器で全処理の時間とBESSの終端状態を記録する。"""
from pathlib import Path
from datetime import datetime
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
MODELS={'aemo':'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306',
        'ercot':'wr_cap77wd_ercotplan_6station_resbid_tanh_ls717_20261013_023145',
        'pjm':'wr_cap77wd_pjmplan_6station_resbid_tanh_g04l15_20261013_081409'}


def temp():
    p=CONTROL/'temperature_status.json'
    x=json.loads(p.read_text(encoding='utf-8'));x['valid']=x.get('valid') and time.time()-p.stat().st_mtime<90
    return x


def record(stage,**extra):
    p=HERE/'latency_status.json';q=p.with_suffix('.tmp')
    q.write_text(json.dumps({'time':datetime.now().astimezone().isoformat(),'stage':stage,**extra}),encoding='utf-8');q.replace(p)


def event(kind,**extra):
    with (CONTROL/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':kind,'experiment':'control_latency_audit',**extra})+'\n')


def main():
    for task,model in MODELS.items():
        for force in (False,True):
            mode='force' if force else 'raw'
            done=HERE/f'bess6_{task}_{mode}_v2.DONE'
            if done.exists():continue
            while True:
                x=temp()
                if (CONTROL/'PAUSE_AUTONOMY').exists() or not (x.get('valid') and not x.get('hot_reason') and x['cpu_package_c']<80):
                    record('waiting',task=task,mode=mode);time.sleep(30);continue
                args=[sys.executable,'-X','utf8',str(HERE/'eval_bess_checked.py'),'--task',task,
                      '--model',str(ROOT/'archive'/model),'--workers','1','--sizes','1000']
                if force:args.append('--force')
                with (HERE/f'{task}_{mode}.log').open('a',encoding='utf-8') as f:
                    child=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                affinity=ctypes.windll.kernel32.SetProcessAffinityMask
                affinity.argtypes=[ctypes.c_void_p,ctypes.c_size_t];affinity.restype=ctypes.c_int
                if not affinity(int(child._handle),(1<<20)|(1<<21)):
                    child.kill();raise OSError('cannot set affinity')
                record('running',task=task,mode=mode,pid=child.pid)
                paused=False
                while child.poll() is None:
                    x=temp()
                    if not x.get('valid') or x.get('hot_reason') or (CONTROL/'PAUSE_AUTONOMY').exists():
                        subprocess.run([str(TASKKILL),'/PID',str(child.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                        paused=True;event('temperature_pause',cpu_c=x.get('cpu_package_c'));break
                    time.sleep(5)
                if paused:time.sleep(30);continue
                if child.returncode:raise RuntimeError(f'{task}/{mode} failed')
                break
    summary={}
    for task in MODELS:
        for mode in ('raw','force'):
            rows=[json.loads(x) for x in (HERE/f'bess6_{task}_{mode}_v2.jsonl').read_text(encoding='utf-8').splitlines()]
            if len(rows)!=40:raise RuntimeError('expected40 sets')
            old=[json.loads(x) for x in (CONTROL/f'bess6_{task}_{mode}_v2.jsonl').read_text(encoding='utf-8').splitlines()]
            old={(r['day'],r['command'],r['ev_seed']):r for r in old if r['bess_kw']==1000}
            for r in rows:
                base=old[(r['day'],r['command'],r['ev_seed'])]
                for field in ('soc_hit_pct','pre_blocks_failed','post_blocks_failed','post_step_pass_pct','post_mae_kw'):
                    if abs(r[field]-base[field])>1e-6:raise RuntimeError(f'instrumentation changed {task} {mode} {field}')
            ends=[r['bess_energy_trace_kwh'][-1]-r['bess_energy_trace_kwh'][0] for r in rows]
            summary[task+'_'+mode]={'sets':40,'outcomes_match_previous':True,
                'mean_episode_full_step_p95_ms':sum(r['full_simulation_step_ms_p95'] for r in rows)/40,
                'max_full_simulation_step_ms':max(r['full_simulation_step_ms_max'] for r in rows),
                'minimum_end_minus_start_energy_kwh':min(ends),
                'episodes_ending_below_start':sum(x<-1e-6 for x in ends)}
    (HERE/'latency_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    record('complete');event('experiment_complete',summary=summary)


if __name__=='__main__':
    try:main()
    except Exception as exc:record('failed',error=str(exc));event('experiment_failed',error=str(exc));raise
