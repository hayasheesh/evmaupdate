"""温度停止したWindowsの本線を、保存した状態から再開する。"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODELS={'ercot':'wr_cap77wd_ercotplan_6station_resbid_tanh_ls717_20261013_023145',
        'pjm':'wr_cap77wd_pjmplan_6station_resbid_tanh_g04l15_20261013_081409'}


def main():
    temp=json.loads((HERE/'temperature_status.json').read_text(encoding='utf-8'))
    if not temp.get('valid') or not temp.get('cool_for_restart'):
        print('pending cooling');return
    ps="[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    ps_exe=Path(os.environ['SystemRoot'])/'System32'/'WindowsPowerShell'/'v1.0'/'powershell.exe'
    processes=json.loads(subprocess.check_output([str(ps_exe),'-NoProfile','-Command',ps],text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW) or '[]')
    if isinstance(processes,dict):processes=[processes]
    stage=ROOT/'execute_results'/'resbid_6station_20261012'
    for task,name in MODELS.items():
        if any(name in (p.get('CommandLine') or '') and 'stage_worker.py' in (p.get('CommandLine') or '') for p in processes):continue
        run=ROOT/'archive'/name
        saved=json.loads((run/'resume'/'latest.json').read_text(encoding='utf-8'))
        if saved['completed_training_episode']>=2000:continue
        if saved.get('reason')!='stop-request':
            print(f'hold {task}: not a saved temperature stop');continue
        marker=run/'resume'/'STOP_REQUESTED'
        if marker.exists():marker.rename(marker.with_name('STOP_REQUESTED.temperature_done_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
        env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
        with (stage/'logs'/f'{task}_pretrain.win.stdout.log').open('a',encoding='utf-8') as out, (stage/'logs'/f'{task}_pretrain.win.stderr.log').open('a',encoding='utf-8') as err:
            p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(stage/'stage_worker.py'),task,'pretrain','--resume-run',str(run)],
                               cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        import ctypes
        set_affinity=ctypes.windll.kernel32.SetProcessAffinityMask
        set_affinity.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        set_affinity.restype=ctypes.c_int
        if not set_affinity(int(p._handle),(1<<32)-1):
            raise OSError('training CPU affinity could not be set')
        (HERE/'training_started_at.json').write_text(json.dumps({'time':datetime.now().astimezone().isoformat(),'task':task,'pid':p.pid}),encoding='utf-8')
        with (HERE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':'training_resumed_after_cooling','task':task,'episode':saved['completed_training_episode'],'pid':p.pid})+'\n')
        print(f'resumed {task} at ep{saved["completed_training_episode"]} PID={p.pid}')
        break  # 学習の起動を重ねず、次の温度確認を待つ。


if __name__=='__main__':main()
