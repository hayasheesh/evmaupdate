"""研究室の保存済み20 station ABGをCPUだけで同じ40組で評価する。"""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
JOBS=[('aemo','wr_cap77wd_aemoplan_20station_resbid_abg_20261013_101724',1920),
      ('ercot','wr_cap77wd_ercotplan_20station_resbid_abg_20261013_101830',1680)]


if __name__=='__main__':
    (HERE/'logs').mkdir(parents=True,exist_ok=True)
    for task,model,ep in JOBS:
        out=HERE/f'abg20_{task}_v2'
        if (out/'DONE').exists():continue
        with (HERE/'logs'/f'lab_{task}.log').open('a',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/'run_eval.py'),'--task',task,
                            '--stage-dir',str(ROOT/'execute_results'/'abg_resbid_20station_20261013'),
                            '--run-dir',str(ROOT/'archive'/model),'--episode',str(ep),
                            '--output',str(out),'--workers','2'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
