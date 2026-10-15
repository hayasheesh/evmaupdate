"""温度と保存点を条件に、評価・集計・停止した計算の再開を順に行う。"""
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
LOSS=ROOT/'execute_results'/'bid_loss_6station_20261013'
LOG=HERE/'logs'/'continue_work.log'
PS_EXE=Path(os.environ['SystemRoot'])/'System32'/'WindowsPowerShell'/'v1.0'/'powershell.exe'
TASKKILL=Path(os.environ['SystemRoot'])/'System32'/'taskkill.exe'


def event(name,**data):
    with (HERE/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':name,**data},ensure_ascii=False)+'\n')


def processes():
    code="[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    data=json.loads(subprocess.check_output([str(PS_EXE),'-NoProfile','-Command',code],text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW) or '[]')
    return [data] if isinstance(data,dict) else data


def run(args,logname):
    with (HERE/'logs'/logname).open('a',encoding='utf-8') as f:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u',*map(str,args)],cwd=ROOT,
                           stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    # 起動直後のライブラリ初期化も2論理CPUに制限する。子プロセスはこの制限を継承する。
    import ctypes
    set_affinity=ctypes.windll.kernel32.SetProcessAffinityMask
    set_affinity.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    set_affinity.restype=ctypes.c_int
    if not set_affinity(int(p._handle),(1<<18)|(1<<19)):
        raise OSError('CPU job affinity could not be set')
    event('cpu_job_started',pid=p.pid,command=list(map(str,args)))
    return p


def loss_complete():
    for task in ('aemo','ercot'):
        for cmd in ('known','unknown'):
            for acc in ('85','91','95'):
                for sel in ('low2max','minmedmax','random3'):
                    f=LOSS/f'result_{task}_{cmd}_{acc}_{sel}.jsonl'
                    if not f.exists():return False
                    text=f.read_text(encoding='utf-8')
                    if not text.endswith('\n') or len(text.splitlines())!=40:return False
    return True


def ready_evaluation():
    from local_queue import JOBS
    return any(not (HERE/(name+'_v2')/'DONE').exists() and
               (ROOT/'archive'/model/'results'/f'TEST{ep}'/f'actor_0_ep{ep}.pth').exists()
               for name,task,stage,model,ep,modes in JOBS)


def pending_evaluations():
    from local_queue import JOBS
    return [name for name,task,stage,model,ep,modes in JOBS if not (HERE/(name+'_v2')/'DONE').exists()]


def main():
    (HERE/'continue_work.pid').write_text(str(os.getpid()),encoding='ascii')
    job=None; job_name=None; loss_job=None; sync_job=None; previous_report=None; hot_active=False; sensor_alert=False; last_sync=0
    while True:
        try:
            if (HERE/'PAUSE_AUTONOMY').exists():
                time.sleep(30);continue
            temp=json.loads((HERE/'temperature_status.json').read_text(encoding='utf-8'))
            age=time.time()-(HERE/'temperature_status.json').stat().st_mtime
            if time.time()-last_sync>300 and (sync_job is None or sync_job.poll() is not None):
                with (HERE/'logs'/'sync_lab.log').open('a',encoding='utf-8') as sync_log:
                    sync_job=subprocess.Popen([sys.executable,'-X','utf8',str(HERE/'sync_lab_results.py')],cwd=ROOT,
                                              stdout=sync_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                last_sync=time.time()
            valid=temp.get('valid') and age<90
            ps=processes()
            from algorithm_watch import check as check_algorithm
            check_algorithm(ps,event)
            cpu_processes=[p for p in ps if any(s in (p.get('CommandLine') or '') for s in ('local_queue.py','loss_bid.py run ','eval_bess.py'))]
            evaluation_busy=any('loss_bid.py run ' not in (p.get('CommandLine') or '') for p in cpu_processes)
            loss_busy=any('loss_bid.py run ' in (p.get('CommandLine') or '') for p in cpu_processes)
            if not valid:
                if not sensor_alert:event('temperature_monitor_unavailable');sensor_alert=True
            else:
                sensor_alert=False
            missing=not valid and (temp.get('sensor_failures',0)>=3 or age>=90)
            if (valid and temp.get('hot_reason')) or missing:
                reason=temp.get('hot_reason') if valid else 'temperature sensor unavailable'
                if not hot_active:event('temperature_pause',reason=reason,cpu_c=temp.get('cpu_package_c'))
                hot_active=True
                for p in cpu_processes:
                    # 途中の入札・評価は、保存済みの組から再開する。
                    subprocess.run([str(TASKKILL),'/PID',str(p['ProcessId']),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                for p in ps:
                    cmd=p.get('CommandLine') or ''
                    if 'stage_worker.py' not in cmd or 'resbid_6station_20261012' not in cmd:continue
                    from resume_after_cooling import MODELS
                    for task,model in MODELS.items():
                        if model in cmd:
                            (ROOT/'archive'/model/'resume'/'STOP_REQUESTED').write_text('temperature stop\n',encoding='utf-8')
                job=None;job_name=None;loss_job=None
            if valid and temp.get('cool_for_restart'):
                from resume_after_cooling import main as resume
                resume()
                if hot_active:event('temperature_cleared',cpu_c=temp['cpu_package_c'])
                hot_active=False
            if job is not None and job.poll() is not None:
                event('cpu_job_finished',job=job_name,exit_code=job.returncode)
                if job.returncode:
                    (HERE/'FAILED').write_text(f'{job_name}: exit {job.returncode}',encoding='utf-8')
                    return
                job=None;job_name=None
            if loss_job is not None and loss_job.poll() is not None:
                event('cpu_job_finished',job='bid_loss_aemo',exit_code=loss_job.returncode)
                if loss_job.returncode:raise RuntimeError('bid-loss calculation failed')
                loss_job=None
            start_marker=HERE/'training_started_at.json'
            training_warming=start_marker.exists() and time.time()-start_marker.stat().st_mtime<180
            can_start=valid and not hot_active and not training_warming and temp['cpu_package_c']<80
            if can_start and not loss_busy and loss_job is None and not loss_complete():
                loss_job=run([LOSS/'loss_bid.py','run','aemo','1'],'loss_aemo_continue.log')
            if can_start and not evaluation_busy and job is None:
                if ready_evaluation():
                    job=run([HERE/'local_queue.py'],'queue_launcher.log');job_name='evaluation_queue'
                else:
                    from resume_after_cooling import MODELS
                    model_aemo='wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306'
                    for task,model in [('aemo',model_aemo),('ercot',MODELS['ercot']),('pjm',MODELS['pjm'])]:
                        if not (ROOT/'archive'/model/'results'/'TEST2000'/'actor_0_ep2000.pth').exists():continue
                        for force in (False,True):
                            mode='force' if force else 'raw'
                            if (HERE/f'bess6_{task}_{mode}_v2.DONE').exists():continue
                            args=[HERE/'eval_bess.py','--task',task,'--model',ROOT/'archive'/model,'--workers','1']
                            if force:args+=['--force']
                            job=run(args,f'bess_{task}_{mode}.log');job_name=f'bess_{task}_{mode}'
                            break
                        if job is not None:break
            stamp=tuple((f.name,f.stat().st_mtime_ns) for f in HERE.glob('*_v2/DONE'))
            bess_stamp=tuple((f.name,len(f.read_text(encoding='utf-8').splitlines())//40) for f in HERE.glob('bess6_*_v2.jsonl'))
            key=(stamp,loss_complete(),bess_stamp)
            if key!=previous_report and valid and not hot_active:
                subprocess.run([sys.executable,'-X','utf8',str(HERE/'report_controls.py')],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
                subprocess.run([sys.executable,'-X','utf8',str(HERE/'report_bid_loss.py')],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
                subprocess.run([sys.executable,'-X','utf8',str(HERE/'report_bess.py')],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
                event('reports_updated',complete_evaluations=len(stamp),loss_complete=key[1],
                      complete_bess_sweeps=len(list(HERE.glob('bess6_*_v2.DONE'))));previous_report=key
            status={'time':datetime.now().astimezone().isoformat(),'cpu_package_c':temp.get('cpu_package_c'),
                    'temperature_valid':bool(valid),'hot':bool(hot_active),'job':job_name or (('bess_evaluation' if any('eval_bess.py' in (p.get('CommandLine') or '') for p in cpu_processes) else 'evaluation_queue') if evaluation_busy else None),'bid_loss_running':bool(loss_busy or (loss_job and loss_job.poll() is None)),
                    'loss_complete':loss_complete(),'completed_evaluations':len(stamp),
                    'ready_evaluation':ready_evaluation(),'pending_evaluations':pending_evaluations()}
            path=HERE/'work_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
        except json.JSONDecodeError:
            # 書き込み中の記録は、次の周期で読み直す。
            time.sleep(1)
        except Exception as exc:
            event('coordinator_error',error=str(exc))
            (HERE/'FAILED').write_text(str(exc),encoding='utf-8');return
        time.sleep(30)


if __name__=='__main__':
    with LOG.open('a',encoding='utf-8',buffering=1) as f:
        sys.stdout=sys.stderr=f
        main()
