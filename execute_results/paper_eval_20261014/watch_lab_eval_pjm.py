"""20 station PJM の本線（報酬の倍率 大域0.4・局所1.5倍、resbid_tanh_g04l15）が2000回保存されたら、CPUで最終評価する。
watch_lab_eval.py（AEMO・ERCOT）と同じ評価で、出力は main20_pjm_v2。"""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
out=HERE/'main20_pjm_v2'
while not (out/'DONE').exists():
    if (HERE/'PAUSE_AUTONOMY').exists():time.sleep(60);continue
    runs=sorted((ROOT/'archive').glob('wr_cap77wd_pjmplan_20station_resbid_tanh_g04l15_2*'))
    manifest=runs[-1]/'resume'/'latest.json' if runs else None
    if manifest and manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['completed_training_episode']>=2000:
        with (HERE/'logs'/'main20_pjm.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/'run_eval.py'),'--task','pjm',
                            '--stage-dir',str(ROOT/'execute_results'/'resbid_20station_20261014'),
                            '--run-dir',str(runs[-1]),'--episode','2000','--output',str(out),
                            '--workers','1','--modes','marl','marl_force'],stdout=log,stderr=subprocess.STDOUT,check=True)
        break
    time.sleep(300)
