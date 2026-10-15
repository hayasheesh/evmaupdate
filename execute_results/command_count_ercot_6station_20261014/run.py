"""温度監視の下で入札作成と同じ40組のLP評価を順に行う。"""
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


def temp():
    p=CONTROL/'temperature_status.json'
    try:
        x=json.loads(p.read_text(encoding='utf-8'))
        x['valid']=x.get('valid') and time.time()-p.stat().st_mtime<90
        return x
    except (ValueError,OSError):return {'valid':False}


def status(stage,**extra):
    p=HERE/'status.json';q=p.with_suffix('.tmp')
    q.write_text(json.dumps({'time':datetime.now().astimezone().isoformat(),'stage':stage,**extra}),encoding='utf-8');q.replace(p)


def event(name,**extra):
    with (CONTROL/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':name,
                            'experiment':'command_count_ercot_6station',**extra})+'\n')


def main():
    if (HERE/'DONE').exists():return
    for stage in ('build','evaluate'):
        if (HERE/(stage+'.DONE')).exists():continue
        while True:
            x=temp()
            if not (x.get('valid') and not x.get('hot_reason') and x['cpu_package_c']<80) or (CONTROL/'PAUSE_AUTONOMY').exists():
                status('waiting',next_stage=stage);time.sleep(30);continue
            with (HERE/(stage+'.log')).open('a',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(HERE/'measure.py'),stage],cwd=ROOT,
                                       stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            affinity=ctypes.windll.kernel32.SetProcessAffinityMask
            affinity.argtypes=[ctypes.c_void_p,ctypes.c_size_t];affinity.restype=ctypes.c_int
            if not affinity(int(child._handle),(1<<20)|(1<<21)):
                child.kill();raise OSError('cannot set CPU affinity')
            status(stage,pid=child.pid)
            paused=False
            while child.poll() is None:
                x=temp()
                if x.get('hot_reason') or not x.get('valid') or (CONTROL/'PAUSE_AUTONOMY').exists():
                    subprocess.run([str(TASKKILL),'/PID',str(child.pid),'/T','/F'],capture_output=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                    paused=True;event('temperature_pause',cpu_c=x.get('cpu_package_c'),stage=stage);break
                time.sleep(5)
            if paused:time.sleep(30);continue
            if child.returncode:raise RuntimeError(f'{stage} exit {child.returncode}')
            (HERE/(stage+'.DONE')).write_text('complete',encoding='ascii');break
    (HERE/'DONE').write_text('complete',encoding='ascii');status('complete')
    event('experiment_complete',summary=json.loads((HERE/'summary.json').read_text(encoding='utf-8')))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));event('experiment_failed',error=str(exc));raise
