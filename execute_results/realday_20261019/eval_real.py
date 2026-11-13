"""本番のテスト：実データの1日の本当の状況で、本線の行動器、finetune した行動器、control を比べる（Windows と研究室、2026-10-19）。

    python eval_real.py <busy|quiet|average> --actors main:2000,busy:100,control:100 [--bid day|main] [--modes marl,marl_force]
                        [--commands 20] [--workers 12]

paper_eval_20261014/eval_episode.py の run_episode をそのまま使う。EVの1日は、そのパターンの本当の状況（build_real.py が書いた
real_<pattern>.pkl、その日と前の2日の実際のセッション）に固定する（realday.fixed_day）。指令は本線の評価と同じ holdout の指令を
--commands 本（本線の検証用の束の2日目の種）。EVは1通りなので、EVの乱数は0だけ。
  --bid day   入札はその日の入札（build_real.bank_name(pattern, 'validation') の2日目）。fc_<日> は前日の昼の予測で作った入札で、
              本当の状況は同じ日（real_<日>.pkl）
  --bid main  入札は本線の検証用の束の2日目（その日を知らない入札）
行動器は、main が本線の実行、busy・quiet・average・control が realft_worker の最新の実行で、数字は TEST<回> の保存点。
結果は eval/<pattern>_bid_<bid>/result_<行動器>_<回>_<mode>.jsonl（1行1組）。済んだ組は飛ばす。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 学習が GPU を使っているので CPU だけで動かす
HERE = Path(__file__).resolve().parent
PE = HERE.parent / 'paper_eval_20261014'
_STATE = {}


def rd_patterns():
    sys.path.insert(0, str(HERE))
    import realday as rd
    return rd.ALL_PATTERNS + tuple(f'{k}_{p}' for k in ('fc', 'ml') for p in rd.ALL_PATTERNS)


def actor_dir(name: str) -> Path:
    import realft_worker as fw
    if name == 'main':
        return fw.MAIN_RUN
    found = sorted((fw.ROOT / 'archive').glob(f'{fw.model_name(name)}_2*'))
    if not found:
        raise FileNotFoundError(name)
    return found[-1]


def init(pattern: str, bid: str):
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(PE))
    import realday as rd
    import realft_worker as fw
    import build_real
    fw.configure(pattern if bid == 'day' else 'control')
    import eval_episode as E
    test_bank = (rd.world.BANKS / build_real.bank_name(pattern, 'validation')) if bid == 'day' else rd.world.main_bank('aemo', 'validation')
    E._CACHE.update(sw=object(), test_bank=test_bank, model='unused', root=fw.ROOT)
    events, initial = pickle.loads((HERE / f'real_{build_real.base_pattern(pattern)}.pkl').read_bytes())
    rd.fixed_day(events, initial).__enter__()
    _STATE.update(E=E, rd=rd)


def run(job):
    actor, ep, mode, c, n_cmd = job
    os.environ['TABLE3_MODEL_DIR'] = str(actor_dir(actor))
    os.environ['TABLE3_EPISODE'] = str(ep)
    out = _STATE['E'].run_episode(('aemo', mode, _STATE['rd'].DAY_INDEX, c, 0, n_cmd))
    out.pop('traces', None)
    out['actor'] = actor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern', choices=rd_patterns())
    ap.add_argument('--actors', required=True)
    ap.add_argument('--modes', default='marl,marl_force')
    ap.add_argument('--bid', choices=('day', 'main'), default='day')
    ap.add_argument('--commands', type=int, default=20)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    out_dir = HERE / 'eval' / f'{args.pattern}_bid_{args.bid}'
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, files = [], {}
    for item in args.actors.split(','):
        actor, ep = item.split(':')
        for mode in args.modes.split(','):
            path = out_dir / f'result_{actor}_{ep}_{mode}.jsonl'
            files[(actor, int(ep), mode)] = path
            done = {r['command'] for r in map(json.loads, path.open(encoding='utf-8'))} if path.exists() else set()
            jobs += [(actor, int(ep), mode, c, args.commands) for c in range(args.commands) if c not in done]
    mp.set_start_method('spawn', force=True)
    t0 = time.time()
    handles = {k: p.open('a', encoding='utf-8') for k, p in files.items()}
    try:
        with mp.Pool(args.workers, initializer=init, initargs=(args.pattern, args.bid)) as pool:
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
