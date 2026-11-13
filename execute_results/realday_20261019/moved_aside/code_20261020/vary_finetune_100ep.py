"""学習し直しのやり方を変えて比べる（AEMO 6 station、研究室、2026-10-20）。F（multift_ml_<日>）の起動も受け持つ。

    python vary_finetune.py [--episodes 100]

realft_worker.VARIANTS の変え方を ORDER の順に、機械学習の水準の予測の世界の3日（空いた日、5%の日、混んだ日）で、本線の保存した
状態から学習し直す（run_rft.py vary_<変え方>_ml_<日>）。B が一番効いた日、ほとんど効かなかった日、下がった日を選んだ。
評価は20・40・60・80・100回に eval_real.py ml_<日> --actors vary_<変え方>_ml_<日>:<回>（1本ずつ）。
先に、9日すべてで vary_frozen_ml_<日>（更新しない20回）を回し、本線の行動器のその日の予測の世界での成績（TEST20）を取る。

GPU の割り振り：学習1本で約2.55 GB 使うので、GPU で動く学習は TOTAL 本まで（nvidia-smi で数える）。GPU の空きが3 GB 未満なら待つ。
E（multi）が済んだら F の3本（multift_ml_<日>、1000回）を先に起動する。E が1800回を過ぎたら、F の分の枠を空けておく。
記録は logs/vary_finetune.log（段の始めと終わり、区切り "milestone <変え方>_<回>"）と vary_eval.log。
失敗したら FAILED_vary_<名前> を書き、その学習の残りは飛ばす。済んだもの（学習の状態が complete、評価の記録に exit 0）は飛ばす。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import more_days as md  # noqa: E402

LOG = HERE / 'logs' / 'vary_finetune.log'
EVAL_LOG = HERE / 'vary_eval.log'
# 短い回数で高い成績に届くかを先に見る順（2026-10-20 21:50 に組み直し。lrx10 の空いた日と5%の日は200回で始めてある）
ORDER = ('lrx10', 'mix50', 'lrx30', 'mix50utd4', 'utd4', 'lrx100', 'mix50lrx10', 'mix50warm10', 'noise0', 'noisex3', 'lrx3',
         'mix25', 'mix50crit10', 'lrx03')
EVAL_EPS = (20, 40, 60, 80, 100)
DAYS = ('quiet', 'q05', 'busy')
# 本線の行動器を、その日の予測の世界の中間テスト（20回目、5エピソード）で測る。更新しないので行動器は本線のまま。評価はしない
FROZEN_DAYS = ('busy', 'quiet', 'average', 'q05')  # 残りの5日は、短い回数の比較を先にするため取りやめた
F_MODES = ('multift_ml_busy', 'multift_ml_quiet', 'multift_ml_average')
TOTAL = 5
E_RESERVE_FROM = 1800


def log(msg, path=LOG):
    md.log(msg, path)


def fail(name):
    (HERE / f'FAILED_vary_{name}').write_text(datetime.now().isoformat(timespec='seconds') + '\n', encoding='utf-8')


def gpu_procs():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=60).stdout
        return len([x for x in out.splitlines() if x.strip()])
    except Exception:  # noqa: BLE001  数えられなければ空きなしとみなす
        return TOTAL


def e_episodes():
    p = HERE / 'logs' / 'multi_pretrain.stdout.log'
    if not p.exists():
        return 0
    return sum(1 for line in p.open(encoding='utf-8', errors='replace') if line.startswith('train'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100)
    args = ap.parse_args()
    import realft_worker as fw
    assert set(ORDER) <= set(fw.VARIANTS)
    def eval_eps(m):
        """評価する回：20〜100回。200回で始めた実行は200回も。"""
        st = HERE / f'{m}_run_status.json'
        n = args.episodes
        if st.exists():
            a = json.loads(st.read_text(encoding='utf-8')).get('args', [])
            n = int(a[a.index('--episodes') + 1]) if '--episodes' in a else n
        return tuple(e for e in EVAL_EPS if e <= n) + ((n,) if n > EVAL_EPS[-1] else ())
    frozen = [f'vary_frozen_ml_{d}' for d in FROZEN_DAYS]
    modes = frozen + [f'vary_{v}_ml_{d}' for v in ORDER for d in DAYS]
    log(f'start modes={len(modes)} episodes={args.episodes} order={",".join(ORDER)}')
    done, bad, failed = set(), set(), set()
    if EVAL_LOG.exists():
        for line in EVAL_LOG.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ['exit', '0']:
                done.add(parts[1])
    children = {}
    milestones = set()

    def launch(m, episodes):
        log(f'train {m} start')
        children[m] = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'run_rft.py', m, '--episodes', str(episodes)],
                                       cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(120)  # 読み込みで GPU の使用量が上がるのを待ってから次を数える

    while True:
        for m, proc in list(children.items()):
            if proc.poll() is not None:
                del children[m]
                log(f'train {m} exit {proc.returncode}')
        states = {m: md.status(m) for m in modes}
        for m, st in states.items():
            if st not in (None, 'running', 'complete') and m not in failed:
                failed.add(m)
                fail(f'train_{m}')
                log(f'train {m} status {st}')
        e_state = md.status('multi')
        f_wait = [m for m in F_MODES if md.status(m) is None and m not in children]
        # F を先に
        if e_state == 'complete':
            for m in f_wait:
                if gpu_procs() >= TOTAL or md.gpu_free_mib() < md.GPU_FREE_MIB:
                    break
                launch(m, 1000)
            f_wait = [m for m in F_MODES if md.status(m) is None and m not in children]
        reserve = len(f_wait) if (e_state == 'complete' or (e_state == 'running' and e_episodes() >= E_RESERVE_FROM)) else 0
        for m in modes:
            if states[m] is not None or m in failed or m in children:
                continue
            if gpu_procs() + reserve >= TOTAL or md.gpu_free_mib() < md.GPU_FREE_MIB:
                break
            launch(m, 20 if m in frozen else args.episodes)
        # 評価（1本ずつ）
        pending = []
        for m in modes:
            if m in frozen:
                continue
            day = m.split('_ml_', 1)[1]
            for ep in eval_eps(m):
                tag = f'{m}{ep}'
                if tag in done or tag in bad or m in failed:
                    continue
                pending.append(tag)
                if not md.ready(m, ep):
                    continue
                time.sleep(30)
                with (HERE / 'logs' / f'eval_{tag}.log').open('a', encoding='utf-8') as out:
                    code = subprocess.run([sys.executable, '-X', 'utf8', '-u', 'eval_real.py', f'ml_{day}', '--actors', f'{m}:{ep}',
                                           '--workers', '8'], cwd=HERE, stdout=out, stderr=subprocess.STDOUT).returncode
                log(f'{tag} exit {code}', EVAL_LOG)
                pending.remove(tag)
                if code == 0:
                    done.add(tag)
                else:
                    bad.add(tag)
                    fail(tag)
        for v in ORDER:
            for ep in EVAL_EPS:
                tags = [f'vary_{v}_ml_{d}{ep}' for d in DAYS if f'vary_{v}_ml_{d}' not in failed]
                name = f'{v}_{ep}'
                if name not in milestones and tags and all(t in done for t in tags):
                    milestones.add(name)
                    log(f'milestone {name}')
        states = {m: md.status(m) for m in modes}
        f_left = [m for m in F_MODES if md.status(m) in (None, 'running')] if e_state in ('running', 'complete') else []
        if not pending and not f_left and all(st not in (None, 'running') or m in failed for m, st in states.items()):
            break
        time.sleep(60)
    log(f'all done failed={",".join(sorted(failed)) or "none"} bad_evals={",".join(sorted(bad)) or "none"}')


if __name__ == '__main__':
    main()
