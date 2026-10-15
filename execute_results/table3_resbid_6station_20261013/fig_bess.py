"""本線のMARL（ep2000）と中央のルールベースに連系点の蓄電池を足したときの、月20日に換算した不適合の回数（6 station AEMO、40組）。
    python fig_bess.py → fig_bess.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
SRC = [('本線 MARL（ep2000）', HERE / 'result_aemo_marl_bess.jsonl', '#2a78d6'),
       ('ルールベース（中央）', HERE.parent / 'rule_bess_6station_20261011' / 'result_aemo_rule_bess_4cmd_2ev.jsonl', '#1baf7a')]
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, path, color in SRC:
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    sizes = sorted({r['bess_kw'] for r in rows})
    month = [sum(r['post_blocks_failed'] for r in rows if r['bess_kw'] == s) / sum(r['bess_kw'] == s for r in rows) * 20 for s in sizes]
    ax.plot(sizes, month, '-o', color=color, lw=2, ms=5, label=name)
ax.axhline(3, color='#666', lw=1, ls='--')
ax.text(600, 3.3, '月3回で停止', ha='right', va='bottom', fontsize=8, color='#444')
ax.set_yscale('symlog', linthresh=1)
ax.set_ylim(0, 150)
ax.set_yticks([0, 1, 3, 10, 30, 100], ['0', '1', '3', '10', '30', '100'])
ax.set_xlabel('連系点の蓄電池の出力（kW、容量は30分を出せる量）')
ax.set_ylabel('月20日に換算した不適合（回）')
ax.set_title('6 station AEMO、40組：蓄電池の大きさと不適合の回数', fontsize=11, loc='left')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.grid(axis='y', color='#e5e5e5', lw=0.8)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / 'fig_bess.png', dpi=160)
print('wrote fig_bess.png')
