"""研究室の既存20局学習の状態を読む。学習や実験は起動しない。"""
import json
import io
import subprocess
import sys
import tarfile
from pathlib import Path

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
REMOTE = r'''
import csv,json,os,re,subprocess,time
from pathlib import Path
root=Path('/home/hayashi/workspace/EVMALOCALUPDATE')
names={'aemo':'allin_aemoplan_AB_20station_20261007_042318','ercot':'allin_ercotplan_AB_20station_20261007_042436'}
out={'checked_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'runs':{}}
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout.strip()
probe=subprocess.run(['docker','exec','-u','1007:1007','-e','HOME=/home/hayashi','hayashi-gpu','/workspace/.venvs/evma20/bin/python','-c','import torch; print(torch.cuda.is_available())'],capture_output=True,text=True,timeout=30)
out['container_gpu_visible']=probe.returncode==0 and probe.stdout.strip().splitlines()[-1:] == ['True']
if probe.returncode:out['container_gpu_error']=probe.stderr[-2000:]
mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
out['mem_available_gib']=round(int(mem['MemAvailable'].split()[0])/1024**2,2)
for market,name in names.items():
 p=root/'execute_results'/f'allin_20station_{market}_20261007'
 s=json.loads((p/'status.json').read_text()); log=p/'pretrain.stdout.log'
 txt=log.read_text(errors='replace'); trains=re.findall(r'^train(\d+)\s.*$',txt,re.M)
 lines=[v for v in txt.splitlines() if 'summary]' in v and 'TEST' in v]
 run=root/'archive'/name
 hist=json.loads((run/'results/test_history.json').read_text())
 with (run/'results/test_performance_metrics.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 tests=[]
 for i,row in enumerate(rows):
  try:
   tr=float(row['Dispatch_Tracking_Rate_%']);soc=float(row['SoC_Hit_Rate_%'])
   tests.append({'episode':hist['episodes'][i],'tracking_pct':tr,'soc_pct':soc,'sum':tr+soc,'mae_kw':float(row['Raw_Actor_MAE_kW'])})
  except (KeyError,ValueError,IndexError):pass
 # status PIDs belong to the container namespace; use docker exec to check them.
 check=subprocess.run(['docker','exec','hayashi-gpu','sh','-c','; '.join(f'kill -0 {int(s[k])} 2>/dev/null && echo {k}=alive || echo {k}=gone' for k in ('launcher_pid','child_pid') if s.get(k))],capture_output=True,text=True,timeout=15)
 errors={}
 for f in p.glob('*.stderr.log'):
  text=f.read_text(errors='replace')
  if 'Traceback' in text:errors[f.name]={'tracebacks':text.count('Traceback'),'tail':text[-2000:]}
 out['runs'][market]={'run_dir':str(run),'stage':s.get('stage'),'error':s.get('error'),'processes':check.stdout.strip(),'episode':int(trains[-1]) if trains else None,'stdout_age_s':round(time.time()-log.stat().st_mtime),'latest_summary':lines[-1] if lines else None,'test_average_last6':tests[-6:],'stderr_errors':errors}
 health=subprocess.run(['docker','exec','-u','1007:1007','-e','HOME=/home/hayashi','-w','/workspace/EVMALOCALUPDATE','hayashi-gpu','/workspace/.venvs/evma20/bin/python','execute_results/remote_20station_bid_unseen_20261002/health.py',f'/workspace/EVMALOCALUPDATE/archive/{name}','--last-windows','3'],capture_output=True,text=True,timeout=30)
 # Only use the existing health reader's gradient rows; its tracking reconstruction is old.
 out['runs'][market]['gradient_health']=[line for line in health.stdout.splitlines() if re.match(r'^\s*\d+-\d+\s*\|',line) or '[alarm]' in line or 'window |' in line]
 if health.returncode:out['runs'][market]['health_read_error']=health.stderr[-1000:]
print(json.dumps(out,ensure_ascii=False,indent=2))
'''

def main():
    result = subprocess.run([SSH, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'smartgrid-gpu', 'python3 -'], input=REMOTE, text=True,
                            encoding='utf-8', capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    data = json.loads(result.stdout)
    if '--mirror' in sys.argv:
        mirror(data)
    print(json.dumps(data, ensure_ascii=False, indent=2))

def mirror(data):
    """現行2本の図表とTensorBoard記録を写す。学習状態は対象外。"""
    script = r'''
import sys,tarfile
from pathlib import Path
root=Path('/home/hayashi/workspace/EVMALOCALUPDATE/archive')
names=('allin_aemoplan_AB_20station_20261007_042318','allin_ercotplan_AB_20station_20261007_042436')
files=('test_performance_metrics.png','test_performance_metrics.csv','train_performance_metrics.png','train_performance_metrics.csv','test_history.json','episode_rewards_all.png','episode_rewards_all.csv','test_episode_rewards_all.png','test_episode_rewards_all.csv')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in names:
  run=root/name
  paths=[run/'results'/f for f in files]
  for sub in ('performance','runs'):paths.extend((run/sub).glob('events.out.tfevents.*'))
  for p in paths:
   if p.is_file():tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
'''
    proc = subprocess.run([SSH,'-o','BatchMode=yes','-o','ConnectTimeout=15',
                           'smartgrid-gpu','python3 -'],input=script.encode(),
                          capture_output=True,timeout=90)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode('utf-8',errors='replace'))
    archive = Path(__file__).resolve().parents[2] / 'archive'
    allowed = {Path(r['run_dir']).name for r in data['runs'].values()}
    # Validate every entry before writing; reject links and paths outside these runs.
    with tarfile.open(fileobj=io.BytesIO(proc.stdout),mode='r:gz') as tar:
        members=tar.getmembers()
        for member in members:
            path=Path(member.name)
            target=(archive/path).resolve()
            if not member.isfile() or path.is_absolute() or '..' in path.parts or path.parts[0] not in allowed or not target.is_relative_to(archive.resolve()):
                raise ValueError(f'Unexpected mirror path: {member.name}')
        for member in members:
            target=archive/member.name
            target.parent.mkdir(parents=True,exist_ok=True)
            source=tar.extractfile(member)
            target.write_bytes(source.read())
    for name in allowed:
        (archive/name/'LAB_MIRROR.txt').write_text(
            '研究室の図表・CSV・テスト履歴・TensorBoard記録の写し。学習状態は研究室にある。\n'
            f'元: smartgrid-gpu:/home/hayashi/workspace/EVMALOCALUPDATE/archive/{name}\n'
            f'確認時刻: {data["checked_at"]}\n',encoding='utf-8')
    data['mirrored_files']=len(members)

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
