"""本線（ep2000）、前の本線（obsdrop、ep1100）、中央のルールベースを、6 station AEMO の同じ40組で並べる図を2枚作る。

    python figs.py   → fig_band_pass.png（帯の幅ごとの30分コマの合格率）、fig_step_pass.png（組ごとの5分ごとの合格率。P90 の基準90%に線）
前の本線とルールベースは table3_6station_20261012/blocks6/ の結果（コマごとの記録つき）を使う。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'table3_6station_20261012' / 'blocks6'
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
SERIES = [('ルールベース（中央）', OLD / 'result_aemo_rule_4cmd_2ev.jsonl', '#1baf7a'),
          ('前の本線 MARL（ep1100）', OLD / 'result_aemo_marl_4cmd_2ev.jsonl', '#eb6834'),
          ('本線 MARL（ep2000）', HERE / 'result_aemo_marl_4cmd_2ev.jsonl', '#2a78d6')]
EDGES = [0, 10, 15, 20, 30, 1e9]
LABEL = ['10 kW未満', '10〜15', '15〜20', '20〜30', '30 kW以上']


def load(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999')
    ax.spines['bottom'].set_color('#999')
    ax.tick_params(colors='#444')
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)


# 図1：帯の幅ごとの30分コマの合格率
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.26
for j, (name, path, color) in enumerate(SERIES):
    cells = [[0, 0] for _ in LABEL]
    for r in load(path):
        for b in r.get('blocks', []):
            i = next(i for i in range(len(LABEL)) if EDGES[i] <= b['band_kw'] < EDGES[i + 1])
            cells[i][0] += 1
            cells[i][1] += b['pass']
    rate = [100 * p / n if n else np.nan for n, p in cells]
    x = np.arange(len(LABEL)) + (j - 1) * (w + 0.02)
    ax.bar(x, rate, w, color=color, label=name)
    if j == 2:
        for xi, v, (n, _p) in zip(x, rate, cells):
            ax.text(xi, v + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=8, color='#333')
ax.set_xticks(np.arange(len(LABEL)), LABEL)
ax.set_xlabel('コマの帯の幅（約定量の10%）')
ax.set_ylabel('30分コマの合格率（%）')
ax.set_ylim(0, 108)
ax.set_title('6 station AEMO、40組：帯の幅ごとの30分コマの合格率', fontsize=11, loc='left')
ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3)
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'fig_band_pass.png', dpi=160)

# 図2：組ごとの5分ごとの合格率（P90 の基準 90%）
fig, ax = plt.subplots(figsize=(8, 4.2))
for j, (name, path, color) in enumerate(SERIES):
    v = np.sort([r['step_pass_pct'] for r in load(path)])
    below = int(np.sum(v < 90))
    ax.plot(np.arange(1, len(v) + 1), v, '-o', color=color, lw=2, ms=4, label=f'{name}：90%未満 {below}/{len(v)}組')
ax.axhline(90, color='#666', lw=1, ls='--')
ax.text(40.5, 90.3, 'P90の基準 90%', ha='right', va='bottom', fontsize=8, color='#444')
ax.set_xlabel('組（5分ごとの合格率の低い順）')
ax.set_ylabel('5分ごとの合格率（%）')
ax.set_title('6 station AEMO、40組：1日の5分ごとの合格率', fontsize=11, loc='left')
ax.legend(frameon=False, fontsize=9, loc='lower right')
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'fig_step_pass.png', dpi=160)
print('wrote figs')
