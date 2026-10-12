"""検証用入札の帯（評価する step の平均）を、入札の作り方ごとに比べる。

    python band_compare.py <aemo|ercot> <入札フォルダ名> ...
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_worker  # noqa: E402
stage_worker.configure(sys.argv[1])
import numpy as np  # noqa: E402
from Config import EPISODE_STEPS  # noqa: E402
from training.bid_bank import BidBank  # noqa: E402
from training.lower_bid_training import build_fixed_upper_bid_training_episode  # noqa: E402
for name in sys.argv[2:]:
    bank = BidBank(HERE.parents[0] / 'bid_banks' / name)
    tols, steps, award = [], 0, []
    for entry in bank.entries:
        fb = dict(bank.load_entry(entry))
        _t, tol, arr, _ = build_fixed_upper_bid_training_episode(fb, 0)
        tr = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
        tol = np.asarray(tol, float)[:EPISODE_STEPS]
        tols += list(tol[tr]); steps += int(tr.sum())
    t = np.array(tols)
    print(f'{name}: 評価 step {steps}  帯の平均 {t.mean():.1f} kW  中央値 {np.median(t):.1f}  50 kW未満の step {100*np.mean(t<50):.0f}%')
