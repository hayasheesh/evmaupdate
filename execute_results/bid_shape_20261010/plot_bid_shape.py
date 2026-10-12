"""今の学習に使っている入札（定格7.7 kW、平日、7 stationは最低80 kW・余裕8 kW、20 stationは最低250 kW・余裕25 kW）の形を描く。
上げを正、下げを負に、30分コマごとに日ごとの線（薄）と中央値（濃）で描く。学習用25日と検証用5日を合わせる。

    python plot_bid_shape.py <20 stationの入札を写したフォルダ>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
HERE = Path(__file__).resolve().parent
BANKS7 = HERE.parent / 'bid_banks'
BANKS20 = Path(sys.argv[1])


def load(root, prefix, signal):
    up, dn, base, days = [], [], [], []
    for split in ('train_25', 'validation_5'):
        for f in sorted((root / f'{prefix}_{split}_{signal}').glob('days/*/results/submitted_day_ahead_bid.csv')):
            rows = list(csv.DictReader(open(f, encoding='utf-8')))
            up.append([float(r['submitted_up_kw']) for r in rows])
            dn.append([float(r['submitted_down_kw']) for r in rows])
            base.append([float(r['baseline_kw']) for r in rows])
            days.append(f.parents[1].name[4:])
    return np.array(up), np.array(dn), np.array(base), days


panels = [
    ('7 station・AEMO', BANKS7, 'cap77wd_min80_margin8', '7station_128cmd_3ev_aemo_plan_deviation', 80),
    ('7 station・ERCOT', BANKS7, 'cap77wd_min80_margin8', '7station_128cmd_3ev_ercot_plan_deviation', 80),
    ('20 station・AEMO', BANKS20, 'cap77wd_min250_margin25', '20station_128cmd_3ev_aemo_plan_deviation', 250),
    ('20 station・ERCOT', BANKS20, 'cap77wd_min250_margin25', '20station_128cmd_3ev_ercot_plan_deviation', 250),
]
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
x = np.arange(48) * 0.5 + 0.25
summary = []
for ax, (title, root, prefix, signal, minbid) in zip(axes.flat, panels):
    up, dn, base, days = load(root, prefix, signal)
    for u, d in zip(up, dn):
        ax.step(x, u, where='mid', color='tab:red', alpha=0.15, lw=0.8)
        ax.step(x, -d, where='mid', color='tab:blue', alpha=0.15, lw=0.8)
    ax.step(x, np.median(up, 0), where='mid', color='tab:red', lw=2.2, label='上げ（中央値）')
    ax.step(x, -np.median(dn, 0), where='mid', color='tab:blue', lw=2.2, label='下げ（中央値、負で表示）')
    ax.step(x, np.median(base, 0), where='mid', color='0.3', lw=1.5, ls='--', label='基準の電力（中央値）')
    ax.axhline(0, color='k', lw=0.6)
    for y in (minbid, -minbid):
        ax.axhline(y, color='0.5', lw=0.8, ls=':')
    part = ((up > 0) | (dn > 0))
    ax.set_title(f'{title}（{len(days)}日）  入札したコマ：1日平均 {part.sum(1).mean():.1f}／48  '
                 f'上げ {(up > 0).sum(1).mean():.1f}・下げ {(dn > 0).sum(1).mean():.1f}', fontsize=11)
    ax.set_ylabel('kW')
    ax.grid(alpha=0.3)
    summary.append((title, len(days), part.sum(1).mean(), (up > 0).sum(1).mean(), (dn > 0).sum(1).mean(),
                    up[up > 0].mean() if (up > 0).any() else 0, dn[dn > 0].mean() if (dn > 0).any() else 0,
                    up.max(), dn.max()))
for ax in axes[1]:
    ax.set_xlabel('時刻（時）')
    ax.set_xticks(range(0, 25, 3))
axes[0, 0].legend(loc='upper left', fontsize=9)
fig.suptitle('今の入札の形（定格7.7 kW・平日、学習用25日＋検証用5日、薄い線は各日、点線は最低入札量）', fontsize=13)
fig.tight_layout()
out = HERE / 'bid_shape.png'
fig.savefig(out, dpi=110)
print(out)
for s in summary:
    print('%s days=%d blocks=%.1f up=%.1f down=%.1f mean_up_kw=%.1f mean_down_kw=%.1f max_up=%.1f max_down=%.1f' % s)
