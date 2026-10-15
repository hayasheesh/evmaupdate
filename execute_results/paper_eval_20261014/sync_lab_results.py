"""研究室で完了した評価だけをWindowsへ写す。"""
from pathlib import Path
import json
import subprocess

HERE=Path(__file__).resolve().parent
BASE='/home/hayashi/workspace/EVMALOCALUPDATE/execute_results/paper_eval_20261014'
for task in ('aemo','ercot'):
    name=f'main20_{task}_v2';dest=HERE/name
    if (dest/'DONE').exists():continue
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','smartgrid-gpu',f'test -f {BASE}/{name}/DONE'],timeout=20)
    if r.returncode:continue
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ('provenance.json',f'result_{task}_marl_4cmd_2ev.jsonl',f'result_{task}_marl_force_4cmd_2ev.jsonl','DONE'):
        subprocess.run(['scp','-q',f'smartgrid-gpu:{BASE}/{name}/{filename}',str(dest)],timeout=120,check=True)
    print(json.dumps({'copied':name}),flush=True)
