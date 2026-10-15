"""評価の記録（paper_eval_20261014/eval_episode.py の result_*.jsonl）から、PJM の規則の点数をまとめる。

    python report_pjm.py <result.jsonl> [...]

5分ごとの点数は market.pjm_performance.interval_scores（1つの5分に1点、10秒の遅れの許容は表せない）。
  - 40組を合わせた点数の平均と最小、25%未満（支払い0）の5分の割合、点数で重みを付けた支払いの割合
  - 同じ指令の番号と EV の乱数の5日を日付の順につなぎ、100時間の移動平均の最小と、40%を保てたか
日本の規則の数値（30分コマの合格率、帯に入った5分の割合）も並べる。
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from market.pjm_performance import (  # noqa: E402
    PAYMENT_FLOOR,
    QUALIFICATION_FLOOR,
    areg_by_step,
    interval_scores,
    qualification_summary,
    score_summary,
)

STEPS_PER_BLOCK = 6


def row_scores(row: dict) -> tuple[np.ndarray, np.ndarray]:
    traces = row['traces']
    residual = np.asarray(traces['residual_kw'], dtype=float)
    target = np.asarray(traces['target_kw'], dtype=float)
    baseline = np.repeat(np.asarray(traces['baseline_kw'], dtype=float), STEPS_PER_BLOCK)[: target.size]
    regulation = target - baseline
    areg = areg_by_step(traces['up_kw'], traces['down_kw'], regulation, steps_per_block=STEPS_PER_BLOCK)
    scores = interval_scores(residual, regulation, areg, np.asarray(traces['tracking'], dtype=bool))
    return scores, areg


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    all_scores, all_areg = [], []
    chains: dict[tuple, list] = defaultdict(list)
    per_set = []
    for row in sorted(rows, key=lambda r: (r['command'], r['ev_seed'], r['day'])):
        scores, areg = row_scores(row)
        all_scores.append(scores)
        all_areg.append(areg)
        chains[(row['command'], row['ev_seed'])].append(scores)
        per_set.append({'day': row['day'], 'command': row['command'], 'ev_seed': row['ev_seed'],
                        **score_summary(scores, areg),
                        'block_pass_pct': row.get('block_pass_pct'), 'step_pass_pct': row.get('step_pass_pct')})
    pooled = score_summary(np.concatenate(all_scores), np.concatenate(all_areg))
    qualification = {f'command{c}_ev{e}': qualification_summary(np.concatenate(parts))
                     for (c, e), parts in sorted(chains.items())}
    lows = [q['min_historic_score'] for q in qualification.values() if q['window_full']]
    return {
        'file': str(path), 'sets': len(rows),
        'pooled': pooled,
        'sets_with_any_unpaid_interval': int(sum(s['share_below_payment_floor'] > 0 for s in per_set)),
        'qualification_chains': len(qualification),
        'chains_with_full_window': len(lows),
        'chains_qualified': int(sum(q['qualified'] is True for q in qualification.values())),
        'min_historic_score': float(min(lows)) if lows else float('nan'),
        'japan_block_pass_pct_mean': float(np.nanmean([s['block_pass_pct'] for s in per_set
                                                        if s['block_pass_pct'] is not None])),
        'payment_floor': PAYMENT_FLOOR, 'qualification_floor': QUALIFICATION_FLOOR,
        'per_set': per_set, 'qualification': qualification,
    }


def main():
    for arg in sys.argv[1:]:
        out = summarize(Path(arg))
        brief = {k: v for k, v in out.items() if k not in ('per_set', 'qualification')}
        print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
        Path(arg).with_suffix('.pjm_score.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
