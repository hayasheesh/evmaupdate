"""固定した入札・40組・保存点で行動器を評価し、5分の誤差を記録する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--stage-dir', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--episode', required=True, type=int)
    ap.add_argument('--output', required=True)
    ap.add_argument('--modes', nargs='+', default=['marl'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    stage = Path(args.stage_dir).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = list((run / 'results' / f'TEST{args.episode}').glob(f'actor_*_ep{args.episode}.pth'))
    if not checkpoints:
        raise RuntimeError('actor checkpoint is missing')
    provenance = {'task': args.task, 'run': run.name, 'episode': args.episode,
                  'stage_dir': str(stage), 'evaluation_sha256': hashlib.sha256((HERE/'eval_episode.py').read_bytes()).hexdigest(),
                  'actors_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints},
                  'sets': {'days': 5, 'commands': 4, 'ev_seeds': 2,
                           'seed_formula': '4000000 + 1000*day + 10*command + ev_seed'}}
    meta = out / 'provenance.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf-8')) != provenance:
        raise RuntimeError('existing results use different evaluation inputs')
    meta.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    env = dict(os.environ, TABLE3_SW_DIR=str(stage), TABLE3_MODEL_DIR=str(run),
               TABLE3_OUT_DIR=str(out), TABLE3_EPISODE=str(args.episode), TABLE3_BLOCKS='1',
               CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    for mode in args.modes:
        with (out / f'{mode}.log').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE/'eval_episode.py'),
                            args.task, mode, '4', '2', str(args.workers)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(x) for x in (out/f'result_{args.task}_{mode}_4cmd_2ev.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = {(d,c,e) for d in range(5) for c in range(4) for e in range(2)}
        if len(rows) != 40 or {(r['day'],r['command'],r['ev_seed']) for r in rows} != expected:
            raise RuntimeError('evaluation set is incomplete or duplicated')
        if mode in ('marl','marl_force') and any(r['model'] != run.name or r['episode'] != args.episode for r in rows):
            raise RuntimeError('evaluation used a different checkpoint')
    (out/'DONE').write_text('40 paired sets per mode\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'complete': True}), flush=True)


if __name__ == '__main__':
    main()
