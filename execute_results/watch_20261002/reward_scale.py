"""最初の数エピソードの大域報酬とQの大きさを並べる（読むだけ）。usage: reward_scale.py RUN_DIR [N]"""
import sys
from health import scalars
run = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
d = scalars(run)
tags = ['GlobalCritic/reward_raw_abs_mean', 'GlobalCritic/reward_scale', 'GlobalCritic/reward_term_abs_mean',
        'GlobalCritic/td_target_abs_mean', 'GlobalCritic/current_q_abs_mean', 'Reward/global', 'System/pre_bess_mae_kw']
for t in tags:
    print(f'{t:<40}', ' '.join(f'{d.get(t, {}).get(ep, float("nan")):9.3f}' for ep in range(1, n + 1)))
