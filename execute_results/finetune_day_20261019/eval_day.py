"""finetune の評価：本線の行動器と、その日の世界で続けた行動器（day）、本線の世界で続けた行動器（control）を同じ組で比べる（Windows、2026-10-19）。

    python eval_day.py --actors main:2000,day:1000,control:1000 [--modes marl,marl_force] [--world day] [--bid day]
                       [--ev 10] [--workers 12]

paper_eval_20261014/eval_episode.py の run_episode をそのまま使う。組は、対象の日（検証用の束の2日目、2024-02-01）
× holdout の指令4本（本線の評価と同じ）× EVの乱数 --ev 通り（set_env_seed(4_000_000 + 1000 × 1 + 10 × 指令 + EV)）。
  --world day   EVはその日の世界（dayworld.day_world、kappa と tau は build_day.CHOSEN）から出す。
  --world count EVはその日の到着台数を中心にした世界（countworld.count_world、k は build_count.K）から出す。
  --world main  EVは本線の世界から出す（本線の評価と同じ）。
  --bid day     入札はその日の入札（build_day が作った day_bid.pkl。finetune 用の検証の束の2日目）。
  --bid count   入札は build_count が作った day_bid_count.pkl（finetune 用の検証の束の2日目）。
  --bid main    入札は本線の検証の束の2日目。
行動器は、main が本線の実行（finetune_worker.MAIN_RUN）、day と control が finetune の最新の実行で、数字は TEST<回> の保存点。
観測の正規化は行動器の実行のもの（finetune の2本は本線と同じ値を写してある）。
結果は eval/world_<world>_bid_<bid>/result_<行動器>_<回>_<mode>.jsonl（1行1組）。済んだ組は飛ばす。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 学習が GPU を使っているので CPU だけで動かす
HERE = Path(__file__).resolve().parent
PE = HERE.parent / 'paper_eval_20261014'
DAY = 1
_STATE = {}


def actor_dir(name: str) -> Path:
    import finetune_worker as fw
    if name == 'main':
        return fw.MAIN_RUN
    found = sorted((fw.ROOT / 'archive').glob(f'ftday_aemo6_20240201_{name}_2*'))
    if not found:
        raise FileNotFoundError(name)
    return found[-1]


def init(world_name: str, bid: str):
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(PE))
    import finetune_worker as fw
    import dayworld as dw
    import build_day
    import build_count
    import countworld
    fw.configure({'day': 'day', 'count': 'count'}.get(bid, 'control'))
    import eval_episode as E
    test_bank = {'day': dw.world.BANKS / build_day.fake_bank_name('validation'),
                 'count': dw.world.BANKS / build_count.count_bank_name('validation')}.get(bid) or dw.world.main_bank('aemo', 'validation')
    E._CACHE.update(sw=object(), test_bank=test_bank, model='unused', root=fw.ROOT)
    if world_name == 'day':
        base = dw.target_base()
        dw.day_world(dw.center(base), build_day.CHOSEN['kappa'], build_day.CHOSEN['tau']).__enter__()
    elif world_name == 'count':
        countworld.count_world(countworld.day_counts(dw.center(dw.target_base())), build_count.K).__enter__()
    _STATE.update(E=E)


def run(job):
    actor, ep, mode, c, e = job
    os.environ['TABLE3_MODEL_DIR'] = str(actor_dir(actor))
    os.environ['TABLE3_EPISODE'] = str(ep)
    out = _STATE['E'].run_episode(('aemo', mode, DAY, c, e, 4))
    out.pop('traces', None)
    out['actor'] = actor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actors', required=True)
    ap.add_argument('--modes', default='marl,marl_force')
    ap.add_argument('--world', choices=('day', 'count', 'main'), default='day')
    ap.add_argument('--bid', choices=('day', 'count', 'main'), default='day')
    ap.add_argument('--ev', type=int, default=10)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    assert args.ev <= 10, 'EV の乱数が次の指令の種と重なる'
    out_dir = HERE / 'eval' / f'world_{args.world}_bid_{args.bid}'
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, files = [], {}
    for item in args.actors.split(','):
        actor, ep = item.split(':')
        for mode in args.modes.split(','):
            path = out_dir / f'result_{actor}_{ep}_{mode}.jsonl'
            files[(actor, int(ep), mode)] = path
            done = {(r['command'], r['ev_seed']) for r in map(json.loads, path.open(encoding='utf-8'))} if path.exists() else set()
            jobs += [(actor, int(ep), mode, c, e) for c in range(4) for e in range(args.ev) if (c, e) not in done]
    mp.set_start_method('spawn', force=True)
    t0 = time.time()
    handles = {k: p.open('a', encoding='utf-8') for k, p in files.items()}
    try:
        with mp.Pool(args.workers, initializer=init, initargs=(args.world, args.bid)) as pool:
            for r in pool.imap_unordered(run, jobs):
                f = handles[(r['actor'], int(r['episode']), r['mode'])]  # 読み込んだ保存点が頼んだ回と違えば、ここで止まる
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
    finally:
        for f in handles.values():
            f.close()
    print(f'{len(jobs)} sets {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
