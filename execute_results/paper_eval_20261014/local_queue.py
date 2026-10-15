"""保存点がそろった評価を順に実行する。学習と入札データは変更しない。"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JOBS = [
    ('main6_aemo', 'aemo', 'resbid_6station_20261012', 'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306', 2000, ['marl']),
    ('main6_aemo_repeat1', 'aemo', 'resbid_6station_20261012', 'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306', 2000, ['marl']),
    ('abg6_aemo', 'aemo', 'abg_resbid_6station_20261013', 'wr_cap77wd_aemoplan_6station_resbid_abg_20261013_114010', 2000, ['marl']),
    ('abg6_ercot', 'ercot', 'abg_resbid_6station_20261013', 'wr_cap77wd_ercotplan_6station_resbid_abg_20261013_114104', 1600, ['marl']),
    ('rule6_aemo', 'aemo', 'resbid_6station_20261012', 'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306', 2000, ['rule']),
    ('rule6_ercot', 'ercot', 'resbid_6station_20261012', 'wr_cap77wd_ercotplan_6station_resbid_tanh_ls717_20261013_023145', 2000, ['rule']),
    ('main6_ercot', 'ercot', 'resbid_6station_20261012', 'wr_cap77wd_ercotplan_6station_resbid_tanh_ls717_20261013_023145', 2000, ['marl','marl_force']),
    ('main6_pjm', 'pjm', 'resbid_6station_20261012', 'wr_cap77wd_pjmplan_6station_resbid_tanh_g04l15_20261013_081409', 2000, ['marl','marl_force']),
    ('rule6_pjm', 'pjm', 'resbid_6station_20261012', 'wr_cap77wd_pjmplan_6station_resbid_tanh_g04l15_20261013_081409', 2000, ['rule']),
]


def main():
    for name, task, stage, model, ep, modes in JOBS:
        out = HERE / (name+'_v2')
        if (out/'DONE').exists():
            continue
        run = ROOT/'archive'/model
        # rule uses fixed bids only; a model checkpoint is still required by the launcher.
        if not (run/'results'/f'TEST{ep}'/f'actor_0_ep{ep}.pth').exists():
            print(f'pending {name} TEST{ep}', flush=True)
            continue
        print(f'start {name}', flush=True)
        subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/'run_eval.py'),
                        '--task',task,'--stage-dir',str(ROOT/'execute_results'/stage),
                        '--run-dir',str(run),'--episode',str(ep),'--output',str(out),
                        '--workers','1','--modes',*modes], cwd=ROOT, check=True)
    print('ready evaluations complete', flush=True)


if __name__ == '__main__':
    with (HERE/'logs'/'local_queue.log').open('a',encoding='utf-8',buffering=1) as log:
        sys.stdout = sys.stderr = log
        main()
