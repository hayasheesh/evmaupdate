"""論文のひな形（LaTeX 版）の図を、execute_results の記録から作る。

    python docs/paper/make_figs.py

出力は docs/paper/figs/*.png。どの図も、6 station の検証用の40組（平日5日×指令4本×EVの乱数2通り）か、
同じ5日の未見通過率（指令128本×EV3通り）の記録から作る。MARL の値は学習途中の保存点のもの。
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ER = ROOT / 'execute_results'
OUT = Path(__file__).resolve().parent / 'figs'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'MS Gothic'], 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200})
C = {'rule': '#2b6cb0', 'marl': '#dd6b20', 'marl_force': '#c05621', 'liu': '#6b46c1', 'mix': '#38a169', 'lp': '#4a5568'}
JP_LIMIT = 100 * (1 - 3 / 960)  # 月3回の不適合、平日20日×48コマ


def jl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)


def fig_liu():
    """Liu/Lyu の再現：SoC達成率の分布と、30分コマの不合格の数。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    labels = ['Liu/Lyuの\n入札と配分', '本研究の入札と\nLiu/Lyuの配分', '本研究の入札と\n中央のルールベース']
    files = ['theirs_force0', 'fixed_force0', 'rule_force0']
    colors = [C['liu'], C['mix'], C['rule']]
    for ti, task in enumerate(('aemo', 'ercot')):
        data = [[r['soc_hit_pct'] for r in jl(ER / f'liu_lyu_2025_port_20261011/result_{task}_{f}_4cmd_2ev.jsonl')] for f in files]
        pos = np.arange(3) + (ti - 0.5) * 0.36
        bp = axes[0].boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True,
                             flierprops={'markersize': 2})
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.55 if task == 'aemo' else 0.95)
        fails = [np.mean([r['blocks_failed'] for r in jl(ER / f'liu_lyu_2025_port_20261011/result_{task}_{f}_4cmd_2ev.jsonl')]) for f in files]
        axes[1].bar(np.arange(3) + (ti - 0.5) * 0.36, fails, width=0.34, color=colors,
                    alpha=0.55 if task == 'aemo' else 0.95, label=task.upper())
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel('SoC達成率（%、1組ごと）')
    axes[0].set_title('(a) 出発時のSoC達成率（左：AEMO、右：ERCOT）', fontsize=8)
    axes[1].axhline(3 / 20, color='k', lw=0.8, ls='--')
    axes[1].text(2.45, 3 / 20 + 0.03, '月3回÷平日20日', fontsize=6, ha='right')
    axes[1].set_ylabel('1日の不合格のコマ（平均）')
    axes[1].set_title('(b) 30分コマの不合格（淡色：AEMO、濃色：ERCOT）', fontsize=8)
    save(fig, 'fig_liu.png')


def fig_information():
    """前日に知っている情報と入札量。"""
    rows = {}
    for task in ('aemo', 'ercot'):
        for mode in ('cmd', 'ev', 'oracle'):
            r = jl(ER / f'info_bid_6station_20261012/result_{task}_{mode}.jsonl')
            rows[(task, mode)] = (np.mean([x['capacity_kw_block'] for x in r]), np.mean([x['main_capacity_kw_block'] for x in r]),
                                  sum(x['feasible'] is True for x in r), len(r))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    names = ['なし（本線）', '当日の指令だけ', '当日のEVだけ', '当日の指令とEV（上限）']
    for ti, task in enumerate(('aemo', 'ercot')):
        main = rows[(task, 'cmd')][1]
        vals = [main] + [rows[(task, m)][0] for m in ('cmd', 'ev', 'oracle')]
        x = np.arange(4) + (ti - 0.5) * 0.38
        bars = ax.bar(x, vals, width=0.36, color=[C['lp'], C['liu'], C['mix'], C['rule']], alpha=0.55 if task == 'aemo' else 0.95)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 250, f'{v / main:.2f}', ha='center', fontsize=6.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel('入札量（kW・コマ／日）')
    ax.set_title('前日に知っている情報と入札量（淡色：AEMO、濃色：ERCOT。数字は本線との比）', fontsize=8)
    ax.set_ylim(0, 26000)
    save(fig, 'fig_information.png')


def fig_controllers():
    """同じ40組での制御器ごとの30分コマの合格率。"""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    groups = [('6', 'aemo', ER / 'table3_6station_20261012'), ('6', 'ercot', ER / 'table3_6station_20261012'),
              ('20', 'aemo', ER / 'table3_6station_20261012/lab20'), ('20', 'ercot', ER / 'table3_6station_20261012/lab20')]
    modes = [('rule', '中央のルールベース', C['rule']), ('marl', '分散MARL（行動器だけ）', C['marl']),
             ('marl_force', '分散MARL＋強制充電', '#f6ad55')]
    for gi, (n, task, d) in enumerate(groups):
        for mi, (m, label, col) in enumerate(modes):
            r = jl(d / f'result_{task}_{m}_4cmd_2ev.jsonl')
            v = 100 * (1 - sum(x['blocks_failed'] for x in r) / sum(x['blocks_assessed'] for x in r))
            ax.bar(gi + (mi - 1) * 0.27, v, width=0.25, color=col, label=label if gi == 0 else None)
            ax.text(gi + (mi - 1) * 0.27, v + 0.6, f'{v:.1f}', ha='center', fontsize=6)
    ax.axhline(JP_LIMIT, color='k', lw=0.8, ls='--')
    ax.text(3.45, JP_LIMIT + 0.8, f'日本の二次②で停止を避ける水準 {JP_LIMIT:.2f}%', fontsize=6.5, ha='right')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'{n} station {t.upper()}' for n, t, _ in groups])
    ax.set_ylabel('30分コマの合格率（%）')
    ax.set_ylim(55, 104)
    ax.legend(fontsize=7, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.13), frameon=False)
    save(fig, 'fig_controllers.png')


def fig_band():
    """帯の幅ごとの30分コマの合格率（6 station、MARL は ep1100）。"""
    edges = [0, 10, 15, 20, 30, 1e9]
    centers = ['<10', '10–15', '15–20', '20–30', '≥30']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, task in zip(axes, ('aemo', 'ercot')):
        for m, label, col in (('rule', '中央のルールベース', C['rule']), ('marl', '分散MARL', C['marl'])):
            cells = [[0, 0] for _ in centers]
            for r in jl(ER / f'table3_6station_20261012/blocks6/result_{task}_{m}_4cmd_2ev.jsonl'):
                for b in r['blocks']:
                    i = next(i for i in range(5) if edges[i] <= b['band_kw'] < edges[i + 1])
                    cells[i][0] += 1
                    cells[i][1] += b['pass']
            ax.plot(range(5), [100 * p / n for n, p in cells], marker='o', color=col, label=label)
        ax.set_xticks(range(5))
        ax.set_xticklabels(centers)
        ax.set_xlabel('帯の幅（kW、約定量の10%）')
        ax.set_title(f'6 station {task.upper()}', fontsize=8)
        ax.axvspan(-0.3, 0.5, color='#e2e8f0', alpha=0.6, lw=0)
    axes[0].text(-0.25, 84, '最低入札量\n70 kW の帯\n＝7 kW', fontsize=6)
    axes[0].set_ylabel('30分コマの合格率（%）')
    axes[0].legend(fontsize=7, loc='lower right')
    save(fig, 'fig_band.png')


def fig_ev_selection():
    """EVの実現の選び方と設計用の指令の本数：入札量と未見通過率。"""
    vol = {('6', 'aemo'): {'main': 15749, 'minmedmax': 18401, '256': 15655}, ('6', 'ercot'): {'main': 16220, 'minmedmax': 18825, '256': 16210},
           ('20', 'aemo'): {'main': 65011, 'minmedmax': 67604, '256': 64848}, ('20', 'ercot'): {'main': 66090, 'minmedmax': 68721, '256': 66155}}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    marks = {'main': ('o', '本線（接続の少ない2通り＋最多1通り、指令128本）'), 'minmedmax': ('s', 'SoCの需要の最小・中央・最大'),
             '256': ('^', '本線のEV、指令256本')}
    for ax, n, sub in ((axes[0], '6', ''), (axes[1], '20', 'lab20/')):
        for task, col in (('aemo', C['rule']), ('ercot', C['marl'])):
            summ = json.loads((ER / f'table3_6station_20261012/{sub}unseen_{task}/summary.json').read_text(encoding='utf-8'))
            for name, v in summ.items():
                key = 'minmedmax' if 'minmedmax' in name else ('256' if '256cmd' in name else 'main')
                rate = 100 * v['feasible'] / v['trials']
                mk, label = marks[key]
                ax.scatter(vol[(n, task)][key] / vol[(n, task)]['main'], rate, marker=mk, color=col, s=28,
                           label=f'{task.upper()}：{label}' if n == '6' else None)
        ax.set_xlabel('入札量（本線との比）')
        ax.set_title(f'{n} station', fontsize=8)
    axes[0].set_ylabel('未見通過率（%）')
    fig.legend(fontsize=6, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.18))
    save(fig, 'fig_ev_selection.png')


def fig_abg():
    """大域批評家だけの学習（ABG）と本線：学習中のテストの追従率とSoC達成率。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, task in zip(axes, ('aemo', 'ercot')):
        for name, label, col in ((f'wr_obsdrop_cap77wd_{task}plan_6station_gs00625', '本線（局所＋大域）', C['rule']),
                                 (f'wr_obsdrop_cap77wd_{task}plan_6station_abg', 'ABG（大域だけ）', C['marl'])):
            d = sorted((ROOT / 'archive').glob(f'{name}_2*'))[-1]
            rows = list(csv.DictReader(open(d / 'results' / 'test_performance_metrics.csv', encoding='utf-8')))
            ep = np.array([int(r['Episode']) * 20 for r in rows])
            for key, ls in (('Dispatch_Tracking_Rate_%', '-'), ('SoC_Hit_Rate_%', ':')):
                y = np.array([float(r[key]) for r in rows])
                y = np.convolve(y, np.ones(5) / 5, mode='valid')
                ax.plot(ep[4:], y, ls=ls, color=col, label=f'{label}・{"追従率" if ls == "-" else "SoC達成率"}')
        ax.set_xlabel('学習のエピソード')
        ax.set_title(f'6 station {task.upper()}（100回の移動平均）', fontsize=8)
    axes[0].set_ylabel('%')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False)
    save(fig, 'fig_abg.png')


def fig_p90():
    """日ごとの5分ごとの合格率の分布（P90 の基準90%）。"""
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    items = [('6', 'aemo', 'table3_6station_20261012'), ('6', 'ercot', 'table3_6station_20261012'),
             ('20', 'aemo', 'table3_6station_20261012/lab20'), ('20', 'ercot', 'table3_6station_20261012/lab20')]
    for gi, (n, task, d) in enumerate(items):
        for mi, (m, col) in enumerate((('rule', C['rule']), ('marl', C['marl']))):
            v = [r['step_pass_pct'] for r in jl(ER / d / f'result_{task}_{m}_4cmd_2ev.jsonl')]
            bp = ax.boxplot([v], positions=[gi + (mi - 0.5) * 0.36], widths=0.3, patch_artist=True, flierprops={'markersize': 2},
                            medianprops={'color': 'k'})
            bp['boxes'][0].set_facecolor(col)
            bp['boxes'][0].set_alpha(0.8)
    ax.axhline(90, color='k', lw=0.8, ls='--')
    ax.text(1.5, 91, 'P90（90%）', fontsize=6.5, ha='center')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'{n} station {t.upper()}' for n, t, _ in items])
    ax.set_ylabel('5分ごとの合格率（%、1組ごと）')
    ax.plot([], [], color=C['rule'], lw=6, label='中央のルールベース')
    ax.plot([], [], color=C['marl'], lw=6, label='分散MARL（行動器だけ）')
    ax.legend(fontsize=7, loc='lower left')
    save(fig, 'fig_p90.png')


def fig_bess():
    """中央のルールベースに蓄電池を足したときの30分コマの合格率。"""
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for task, col in (('aemo', C['rule']), ('ercot', C['marl'])):
        by = {}
        for r in jl(ER / f'rule_bess_6station_20261011/result_{task}_rule_bess_4cmd_2ev.jsonl'):
            by.setdefault(r['bess_kw'], []).append(r)
        kw = sorted(by)
        v = [100 * (1 - sum(x['post_blocks_failed'] for x in by[k]) / sum(x['post_blocks_assessed'] for x in by[k])) for k in kw]
        # 0 kW は蓄電池なしの中央のルールベース（同じ40組）。
        base = jl(ER / f'table3_6station_20261012/result_{task}_rule_4cmd_2ev.jsonl')
        kw = [0.0] + kw
        v = [100 * (1 - sum(x['blocks_failed'] for x in base) / sum(x['blocks_assessed'] for x in base))] + v
        ax.plot(kw, v, marker='o', ms=3, color=col, label=task.upper())
    ax.axhline(JP_LIMIT, color='k', lw=0.8, ls='--')
    ax.text(600, JP_LIMIT - 0.12, f'{JP_LIMIT:.2f}%', fontsize=6.5, ha='right', va='top')
    ax.set_xlabel('蓄電池の出力（kW、容量は30分分。0は蓄電池なし）')
    ax.set_ylabel('30分コマの合格率（%）')
    ax.legend(fontsize=7)
    save(fig, 'fig_bess.png')


if __name__ == '__main__':
    for f in (fig_liu, fig_information, fig_controllers, fig_band, fig_ev_selection, fig_abg, fig_p90, fig_bess):
        f()
        print('wrote', f.__name__)
