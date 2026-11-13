"""学習し直しのやり方を変えて比べる（AEMO 6 station、研究室、2026-10-20）。F（multift_ml_<日>）の起動も受け持つ。

    python vary_finetune.py

realft_worker.VARIANTS の変え方を、機械学習の水準の予測の世界の3日（空いた日、5%の日、混んだ日）で、本線の保存した状態から
学習し直す（run_rft.py vary_<変え方>_ml_<日>）。B が一番効いた日、ほとんど効かなかった日、下がった日を選んだ。
足したテストの日の B（ml_q05〜ml_q95、300回まで済み）も1000回まで続ける。

仕事の順（build_jobs）：先に試す変え方（HIGH）を100回まで → HIGH を1000回まで → 足した日の B を1000回まで → 後に試す変え方（LOW）を
100回まで → LOW を1000回まで。途中で止まっている実行は、保存した状態（resume/）から --resume-run で続ける。
評価：20・40・60・80回と、100回から1000回まで100回ごとに、eval_real.py ml_<日> --actors <実行>:<回>（1本ずつ）。
止める決まり（利用者の指示「明らかに下降していたら止める」、abg_stop.py と同じく300回で見る）：100回ごとの当日の和が、
300回以上前の最高を更新せず、直近の値が最高より3以上低ければ、pre_train.py --request-stop で止める（vary_stopped.json に残す）。
GPU：学習1本で約2.55 GB 使うので、GPU で動く学習は TOTAL 本まで（nvidia-smi で数える）。GPU の空きが3 GB 未満なら待つ。
E（multi）が済んだら F の3本（1000回）を先に起動する。E が1800回を過ぎたら F の分の枠を空けておく。
記録は logs/vary_finetune.log（始めと終わり、区切り "milestone <変え方>_100"・"milestone <変え方>_end"、止めた実行）と vary_eval.log。
失敗したら FAILED_vary_<名前> を書き、その実行の残りは飛ばす。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import more_days as md  # noqa: E402

LOG = HERE / 'logs' / 'vary_finetune.log'
EVAL_LOG = HERE / 'vary_eval.log'
STOP_FILE = HERE / 'vary_stopped.json'
HIGH = ('lrx10', 'mix50', 'lrx30', 'mix50utd4', 'utd4', 'mix50lrx10', 'mix50warm10')
LOW = ('noise0', 'noisex3', 'lrx3', 'lrx03', 'mix25', 'mix50crit10', 'lrx100')
DAYS = ('quiet', 'q05', 'busy')
NEW_B_DAYS = ('q05', 'q20', 'q35', 'q65', 'q80', 'q95')
SCREEN, TARGET = 100, 1000
EVAL_EPS = (20, 40, 60, 80) + tuple(range(100, TARGET + 1, 100))
F_MODES = ('multift_ml_busy', 'multift_ml_quiet', 'multift_ml_average')
TOTAL = 5
E_RESERVE_FROM = 1800
STOP_WINDOW, STOP_DROP = 300, 3.0


def build_jobs():
    jobs = []
    for group in (HIGH, None, LOW):
        if group is None:
            jobs += [(f'ml_{d}', TARGET) for d in NEW_B_DAYS]
            continue
        jobs += [(f'vary_{v}_ml_{d}', SCREEN) for v in group for d in DAYS]
        jobs += [(f'vary_{v}_ml_{d}', TARGET) for v in group for d in DAYS]
    return jobs


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


def latest_dir(m):
    found = sorted((ROOT / 'archive').glob(f'realday_aemo6_{m}_2*'))
    return found[-1] if found else None


def completed(m):
    d = latest_dir(m)
    p = d / 'resume' / 'latest.json' if d else None
    if not p or not p.exists():
        return 0
    return int(json.loads(p.read_text(encoding='utf-8')).get('completed_training_episode', 0))


def day_of(m):
    return m.split('_ml_', 1)[1] if m.startswith('vary_') else m[len('ml_'):]


def real_sum(m, ep):
    p = HERE / 'eval' / f'ml_{day_of(m)}_bid_day' / f'result_{m}_{ep}_marl.jsonl'
    rows = [json.loads(x) for x in p.open(encoding='utf-8')] if p.exists() else []
    return float(np.mean([r['step_pass_pct'] + r['soc_hit_pct'] for r in rows])) if rows else None


def main():
    jobs = build_jobs()
    modes = list(dict.fromkeys(m for m, _ in jobs))
    target_of = {m: max(t for mm, t in jobs if mm == m) for m in modes}
    log(f'start jobs={len(jobs)} modes={len(modes)} high={",".join(HIGH)} low={",".join(LOW)}')
    done, bad, failed = set(), set(), set()
    if EVAL_LOG.exists():
        for line in EVAL_LOG.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ['exit', '0']:
                done.add(parts[1])
    stopped = json.loads(STOP_FILE.read_text(encoding='utf-8')) if STOP_FILE.exists() else {}
    children = {}
    milestones = set()

    def launch(m, target):
        d = latest_dir(m)
        args = [sys.executable, '-X', 'utf8', '-u', 'run_rft.py', m, '--episodes', str(target)]
        if d is not None and md.status(m) is not None:
            args += ['--resume-run', str(d)]
        log(f'train {m} start target={target}' + (f' resume={d.name} at {completed(m)}' if '--resume-run' in args else ''))
        children[m] = subprocess.Popen(args, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(120)  # 読み込みで GPU の使用量が上がるのを待ってから次を数える

    def job_done(m, target):
        if m in failed or m in stopped:
            return True
        return md.status(m) == 'complete' and completed(m) >= target

    def finished(m):
        return job_done(m, target_of[m])

    while True:
        for m, proc in list(children.items()):
            if proc.poll() is not None:
                del children[m]
                log(f'train {m} exit {proc.returncode} completed={completed(m)}')
        for m in modes:
            st = md.status(m)
            if st not in (None, 'running', 'complete') and m not in failed:
                failed.add(m)
                fail(f'train_{m}')
                log(f'train {m} status {st}')
        # F を先に
        e_state = md.status('multi')
        f_wait = [m for m in F_MODES if md.status(m) is None and m not in children]
        if e_state == 'complete':
            for m in f_wait:
                if gpu_procs() >= TOTAL or md.gpu_free_mib() < md.GPU_FREE_MIB:
                    break
                log(f'train {m} start target=1000')
                children[m] = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'run_rft.py', m, '--episodes', '1000'],
                                               cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(120)
            f_wait = [m for m in F_MODES if md.status(m) is None and m not in children]
        reserve = len(f_wait) if (e_state == 'complete' or (e_state == 'running' and e_episodes() >= E_RESERVE_FROM)) else 0
        # 仕事の順に起動する（同じ実行の次の段は、前の段が済んでから）
        seen = set()
        for m, target in jobs:
            if m in seen or job_done(m, target):
                continue
            seen.add(m)
            if md.status(m) == 'running' or m in children:
                continue
            if gpu_procs() + reserve >= TOTAL or md.gpu_free_mib() < md.GPU_FREE_MIB:
                break
            launch(m, target)
        # 評価（1本ずつ）と止める決まり
        pending = []
        for m in modes:
            if m in failed:
                continue
            for ep in (e for e in EVAL_EPS if e <= target_of[m]):
                tag = f'{m}{ep}'
                if tag in done or tag in bad:
                    continue
                if finished(m) and ep > completed(m):
                    continue
                pending.append(tag)
                if not md.ready(m, ep):
                    continue
                time.sleep(30)
                with (HERE / 'logs' / f'eval_{tag}.log').open('a', encoding='utf-8') as out:
                    code = subprocess.run([sys.executable, '-X', 'utf8', '-u', 'eval_real.py', f'ml_{day_of(m)}', '--actors',
                                           f'{m}:{ep}', '--workers', '8'], cwd=HERE, stdout=out, stderr=subprocess.STDOUT).returncode
                log(f'{tag} exit {code}', EVAL_LOG)
                pending.remove(tag)
                if code != 0:
                    bad.add(tag)
                    fail(tag)
                    continue
                done.add(tag)
                if ep % 100 or ep < 100 + STOP_WINDOW or m in stopped:
                    continue
                vals = {e: real_sum(m, e) for e in EVAL_EPS if f'{m}{e}' in done}
                vals = {e: v for e, v in vals.items() if v is not None}
                best_e = max(vals, key=vals.get)
                last = max(e for e in vals if e % 100 == 0)
                if last - best_e >= STOP_WINDOW and vals[last] <= vals[best_e] - STOP_DROP:
                    stopped[m] = {'best_episode': best_e, 'best': vals[best_e], 'last_episode': last, 'last': vals[last],
                                  'completed_at_request': completed(m), 'at': datetime.now().isoformat(timespec='seconds')}
                    STOP_FILE.write_text(json.dumps(stopped, ensure_ascii=False, indent=1), encoding='utf-8')
                    d = latest_dir(m)
                    if md.status(m) == 'running' and d is not None:
                        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'pre_train.py'), '--request-stop', str(d)],
                                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log(f'stop {m}: best {vals[best_e]:.1f} at {best_e}, {vals[last]:.1f} at {last}')
        # 区切り
        for v in HIGH + LOW:
            ms = [f'vary_{v}_ml_{d}' for d in DAYS]
            if f'{v}_100' not in milestones and all(f'{m}100' in done or m in failed for m in ms):
                milestones.add(f'{v}_100')
                log(f'milestone {v}_100')
            if f'{v}_end' not in milestones and all(finished(m) for m in ms) \
                    and not any(t.startswith(tuple(ms)) for t in pending) and not any(m in children for m in ms):
                milestones.add(f'{v}_end')
                log(f'milestone {v}_end')
        newb = [f'ml_{d}' for d in NEW_B_DAYS]
        if 'newb_end' not in milestones and all(finished(m) for m in newb) \
                and not any(t.startswith(tuple(newb)) for t in pending) and not any(m in children for m in newb):
            milestones.add('newb_end')
            log('milestone newb_end')
        f_left = [m for m in F_MODES if md.status(m) in (None, 'running')] if e_state in ('running', 'complete') else []
        if not pending and not f_left and not children and all(finished(m) for m in modes):
            break
        time.sleep(60)
    log(f'all done failed={",".join(sorted(failed)) or "none"} stopped={",".join(sorted(stopped)) or "none"} '
        f'bad_evals={",".join(sorted(bad)) or "none"}')


if __name__ == '__main__':
    main()
