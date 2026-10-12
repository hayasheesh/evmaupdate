"""result_*.jsonl を条件ごとに集計する。"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for p in sorted(HERE.glob('result_*.jsonl')):
    rs = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    if not rs:
        continue
    f = lambda k: np.array([r[k] for r in rs], float)
    days = sorted(set(r['day'] for r in rs))
    per_day_failed = [np.mean([r['blocks_failed'] for r in rs if r['day'] == d]) for d in days]
    print(f"{p.stem}: n={len(rs)}  step={f('step_pass_pct').mean():.1f}%  block={f('block_pass_pct').mean():.1f}% "
          f"(min {f('block_pass_pct').min():.1f})  failed_blocks/day={f('blocks_failed').mean():.2f} (max {f('blocks_failed').max():.0f})  "
          f"soc={f('soc_hit_pct').mean():.1f}% (min {f('soc_hit_pct').min():.1f})  err/award={f('mean_error_to_award_pct').mean():.2f}%  "
          f"capacity={f('offered_capacity_kw_block').mean():.0f}  slack={f('plan_slack_current_block_kw_mean').mean():.2f}kW")
