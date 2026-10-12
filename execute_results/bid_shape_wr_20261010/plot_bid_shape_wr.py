"""職場と住宅だけの構成の入札（定格7.7 kW、平日、6 stationは最低70 kW・余裕7 kW、20 stationは最低250 kW・余裕25 kW）の形を描く。
上げを正、下げを負に、30分コマごとに日ごとの線（薄）と中央値（濃）で描く。学習用と検証用の、できている日をすべて使う。
比べるために、Boulderの街中3局を含む前の構成（7 station・20 station）の入札の中央値を点線で重ねる。

    python plot_bid_shape_wr.py <20 stationの新しい入札を写したフォルダ> <20 stationの前の入札を写したフォルダ>
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
BANKS = HERE.parent / 'bid_banks'
LAB_NEW = Path(sys.argv[1])
LAB_OLD = Path(sys.argv[2])


def load(root, prefix, signal):
    up, dn, base = [], [], []
    for split in ('train_25', 'validation_5'):
        for f in sorted((root / f'{prefix}_{split}_{signal}').glob('days/*/results/submitted_day_ahead_bid.csv')):
            rows = list(csv.DictReader(open(f, encoding='utf-8')))
            up.append([float(r['submitted_up_kw']) for r in rows])
            dn.append([float(r['submitted_down_kw']) for r in rows])
            base.append([float(r['baseline_kw']) for r in rows])
    return np.array(up), np.array(dn), np.array(base)


panels = [
    ('6 station・AEMO', BANKS, 'wr_cap77wd_min70_margin7', '6station_128cmd_3ev_aemo_plan_deviation', 70,
     BANKS, 'cap77wd_min80_margin8', '7station_128cmd_3ev_aemo_plan_deviation', '7 station'),
    ('6 station・ERCOT', BANKS, 'wr_cap77wd_min70_margin7', '6station_128cmd_3ev_ercot_plan_deviation', 70,
     BANKS, 'cap77wd_min80_margin8', '7station_128cmd_3ev_ercot_plan_deviation', '7 station'),
    ('20 station・AEMO', LAB_NEW, 'wr_cap77wd_min250_margin25', '20station_128cmd_3ev_aemo_plan_deviation', 250,
     LAB_OLD, 'cap77wd_min250_margin25', '20station_128cmd_3ev_aemo_plan_deviation', '20 station'),
    ('20 station・ERCOT', LAB_NEW, 'wr_cap77wd_min250_margin25', '20station_128cmd_3ev_ercot_plan_deviation', 250,
     LAB_OLD, 'cap77wd_min250_margin25', '20station_128cmd_3ev_ercot_plan_deviation', '20 station'),
]
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
x = np.arange(48) * 0.5 + 0.25
summary = []
for ax, (title, root, prefix, signal, minbid, oroot, oprefix, osignal, olabel) in zip(axes.flat, panels):
    up, dn, base = load(root, prefix, signal)
    oup, odn, _ = load(oroot, oprefix, osignal)
    for u, d in zip(up, dn):
        ax.plot(x, u, color='tab:red', alpha=0.15, lw=0.8, drawstyle='steps-mid')
        ax.plot(x, -d, color='tab:blue', alpha=0.15, lw=0.8, drawstyle='steps-mid')
    ax.plot(x, np.median(up, 0), color='tab:red', lw=2.2, drawstyle='steps-mid', label='上げ（中央値）')
    ax.plot(x, -np.median(dn, 0), color='tab:blue', lw=2.2, drawstyle='steps-mid', label='下げ（中央値、負で表示）')
    ax.plot(x, np.median(base, 0), color='0.3', lw=1.5, ls='--', drawstyle='steps-mid', label='基準の電力（中央値）')
    ax.plot(x, np.median(oup, 0), color='tab:red', lw=1.2, ls=':', drawstyle='steps-mid', label=f'前の構成（{olabel}）の上げ・下げの中央値')
    ax.plot(x, -np.median(odn, 0), color='tab:blue', lw=1.2, ls=':', drawstyle='steps-mid')
    ax.axhline(0, color='k', lw=0.6)
    for y in (minbid, -minbid):
        ax.axhline(y, color='0.5', lw=0.8, ls=(0, (1, 3)))
    part = (up > 0) | (dn > 0)
    ax.set_title(f'{title}（{len(up)}日）  入札したコマ：1日平均 {part.sum(1).mean():.1f}／48  '
                 f'上げ {(up > 0).sum(1).mean():.1f}・下げ {(dn > 0).sum(1).mean():.1f}', fontsize=11)
    ax.set_ylabel('kW')
    ax.grid(alpha=0.3)
    ou = oup[oup > 0].mean() if (oup > 0).any() else 0
    od = odn[odn > 0].mean() if (odn > 0).any() else 0
    summary.append((title, len(up), part.sum(1).mean(), (up > 0).sum(1).mean(), (dn > 0).sum(1).mean(),
                    up[up > 0].mean() if (up > 0).any() else 0, dn[dn > 0].mean() if (dn > 0).any() else 0,
                    up.max(), dn.max(), np.median(base, 0).mean(), ou, od))
for ax in axes[1]:
    ax.set_xlabel('時刻（時）')
    ax.set_xticks(range(0, 25, 3))
axes[0, 0].legend(loc='upper left', fontsize=8)
fig.suptitle('職場と住宅だけの構成の入札の形（定格7.7 kW・平日、学習用＋検証用、薄い線は各日、細い点線は最低入札量）', fontsize=13)
fig.tight_layout()
out = HERE / 'bid_shape_wr.png'
fig.savefig(out, dpi=110)
print(out)
for s in summary:
    print('%s days=%d blocks=%.1f up=%.1f down=%.1f mean_up=%.1f mean_down=%.1f max_up=%.1f max_down=%.1f mean_baseline=%.1f old_mean_up=%.1f old_mean_down=%.1f' % s)
