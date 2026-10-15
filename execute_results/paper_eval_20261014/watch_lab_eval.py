"""20 stationの学習が2000回保存されたら、CPUで最終評価する。"""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
done=set()
while len(done)<2:
    if (HERE/'PAUSE_AUTONOMY').exists():time.sleep(60);continue
    for task in ('aemo','ercot'):
        if task in done:continue
        out=HERE/f'main20_{task}_v2'
        if (out/'DONE').exists():done.add(task);continue
        runs=sorted((ROOT/'archive').glob(f'wr_cap77wd_{task}plan_20station_resbid_tanh_ls717_2*'))
        if not runs:continue
        run=runs[-1];manifest=run/'resume'/'latest.json'
        if not manifest.exists():continue
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if saved['completed_training_episode']<2000:continue
        with (HERE/'logs'/f'main20_{task}.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/'run_eval.py'),'--task',task,
                            '--stage-dir',str(ROOT/'execute_results'/'resbid_20station_20261014'),
                            '--run-dir',str(run),'--episode','2000','--output',str(out),
                            '--workers','1','--modes','marl','marl_force'],stdout=log,stderr=subprocess.STDOUT,check=True)
        done.add(task)
    time.sleep(60)
