"""TSG 投稿の体裁で、今ある結果から図を作る（概観用、2026-10-15）。

    python make_figs.py        figs/ に PNG を書く

数値は評価の記録（execute_results/paper_eval_20261014 ほか）から計算する。記録が手元に無いものは
docs/論文の結果と実験計画.md の値を使い、その旨を各関数の注記に書く。まだ測っていない条件は「未測定」と描く。
図の幅は IEEE の1段（3.5 in）と2段抜き（7.16 in）。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / 'execute_results' / 'paper_eval_20261014'
SEL = ROOT / 'execute_results' / 'selection_table_6station_20261015'
OUT = HERE / 'figs'
ONE, TWO = 3.5, 7.16
COLORS = {'MARL': '#2a78d6', 'MARL+強制充電': '#7fb0ea', 'ルールベース': '#1baf7a', 'Liu/Lyu': '#9c64c7',
          'Liu/Lyu配分＋本研究の入札': '#c9a4e3'}

plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 7,
                     'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300})


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def metrics(path: Path) -> dict:
    rs = rows(path)
    failed = sum(r['blocks_failed'] for r in rs)
    total = sum(r['blocks_assessed'] for r in rs)
    return {'block': 100 * (1 - failed / total), 'step': float(np.mean([r['step_pass_pct'] for r in rs])),
            'soc': float(np.mean([r['soc_hit_pct'] for r in rs]))}


def placeholder(ax, text='未測定'):
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color='#888888', fontsize=8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#cccccc')


def fig_final():
    """8条件（6・20 station × AEMO・ERCOT・PJM・GB）。20 station のルールベース AEMO・ERCOT は文書の値。"""
    data = {}
    for st, task, folder, mode, label in [
        (6, 'aemo', 'main6_aemo_v2', 'marl', 'MARL'), (6, 'ercot', 'main6_ercot_v2', 'marl', 'MARL'),
        (6, 'pjm', 'main6_pjm_v2', 'marl', 'MARL'), (6, 'ercot', 'main6_ercot_v2', 'marl_force', 'MARL+強制充電'),
        (6, 'pjm', 'main6_pjm_v2', 'marl_force', 'MARL+強制充電'),
        (6, 'aemo', 'rule6_aemo_v2', 'rule', 'ルールベース'), (6, 'ercot', 'rule6_ercot_v2', 'rule', 'ルールベース'),
        (6, 'pjm', 'rule6_pjm_v2', 'rule', 'ルールベース'),
        (20, 'aemo', 'main20_aemo_v2', 'marl', 'MARL'), (20, 'ercot', 'main20_ercot_v2', 'marl', 'MARL'),
        (20, 'aemo', 'main20_aemo_v2', 'marl_force', 'MARL+強制充電'),
        (20, 'ercot', 'main20_ercot_v2', 'marl_force', 'MARL+強制充電'),
        (20, 'pjm', 'rule20_pjm_v2', 'rule', 'ルールベース')]:
        data[(st, task, label)] = metrics(EVAL / folder / f'result_{task}_{mode}_4cmd_2ev.jsonl')
    # 6 station AEMO の行動器＋強制充電は30分コマの合格率だけ文書にある（80.9%、SoC 100%）
    data[(6, 'aemo', 'MARL+強制充電')] = {'block': 80.9, 'step': np.nan, 'soc': 100.0}
    # 文書の値（論文の結果と実験計画.md 3.1節、4.3節、5.3節）
    data[(20, 'aemo', 'ルールベース')] = {'block': 98.8, 'step': 99.4, 'soc': 100.0}
    data[(20, 'ercot', 'ルールベース')] = {'block': 96.7, 'step': 97.4, 'soc': 100.0}
    data[(6, 'aemo', 'Liu/Lyu')] = {'block': 97.8, 'step': 99.5, 'soc': 89.0}
    data[(6, 'ercot', 'Liu/Lyu')] = {'block': 98.1, 'step': 99.3, 'soc': 88.9}
    data[(6, 'pjm', 'Liu/Lyu')] = metrics(ROOT / 'execute_results' / 'liu_lyu_2025_port_20261011' / 'result_pjm_theirs_force0_4cmd_2ev.jsonl')
    methods = ['MARL', 'MARL+強制充電', 'ルールベース', 'Liu/Lyu']
    keys = [('block', '30分コマ'), ('step', '5分'), ('soc', 'SoC')]
    fig, axes = plt.subplots(2, 4, figsize=(TWO, 3.1))
    for i, st in enumerate((6, 20)):
        for j, (task, title) in enumerate([('aemo', 'AEMO'), ('ercot', 'ERCOT'), ('pjm', 'PJM RegD'), ('gb', 'GB')]):
            ax = axes[i, j]
            ax.set_title(f'{st} station・{title}')
            present = [m for m in methods if (st, task, m) in data]
            if not present:
                placeholder(ax)
                continue
            if (st, task) == (20, 'pjm'):
                ax.text(0.62, 0.12, 'MARLは\n学習中', transform=ax.transAxes, ha='center', fontsize=6, color='#888888')
            x = np.arange(len(keys))
            w = 0.8 / len(methods)
            for k, m in enumerate(methods):
                if (st, task, m) not in data:
                    continue
                vals = [data[(st, task, m)][key] for key, _ in keys]
                ax.bar(x + (k - 1.5) * w, vals, w, color=COLORS[m], label=m)
            ax.set_xticks(x)
            ax.set_xticklabels([name for _, name in keys])
            ax.set_ylim(50, 101)
            ax.axhline(99.69, color='#d03b3b', lw=0.6, ls='--', label='日本の要求（コマ99.69%）' if (i, j) == (0, 0) else None)
            if j == 0:
                ax.set_ylabel('合格率・達成率（%）')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / 'fig_final.png')
    plt.close(fig)


def fig_information():
    """入札時に知る情報と入札量（文書4.2節の表の値）。"""
    labels = ['両方知らない', '指令だけ知る', 'EVだけ知る']
    vals = {'6 st AEMO': [68.6, 73.2, 96.5], '6 st ERCOT': [70.6, 73.2, 97.8], '6 st PJM': [73.1, 73.2, 100.0],
            '20 st AEMO': [83.8, 87.5, 96.7], '20 st ERCOT': [85.2, 87.5, 98.2], '20 st PJM': [87.5, 87.5, 100.0]}
    fig, ax = plt.subplots(figsize=(ONE, 2.1))
    x = np.arange(len(labels))
    w = 0.13
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#7fb0ea', '#f3a77e', '#8fd9bc']
    for k, (name, v) in enumerate(vals.items()):
        ax.bar(x + (k - 2.5) * w, v, w, label=name, color=colors[k])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('両方知る入札量に対する比（%）')
    ax.set_ylim(50, 100)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=5.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_information.png')
    plt.close(fig)


def fig_control():
    """制御の前提と性能（6 station、文書4.3節の値）。先のEVや指令を知る制御は未測定。"""
    conds = ['全情報のLP', '先のEVも知る\nルールベース', '先の指令も知る\n制御', '今だけ見る\nルールベース', '分散MARL']
    full_day = {'AEMO': [39, None, None, 30, 9], 'ERCOT': [40, None, None, 32, 7], 'PJM': [40, None, None, 20, 0]}
    fig, ax = plt.subplots(figsize=(ONE, 1.8))
    x = np.arange(len(conds))
    w = 0.26
    for k, (task, vals) in enumerate(full_day.items()):
        for i, v in enumerate(vals):
            if v is None:
                ax.text(x[i] + (k - 1) * w, 1, '未', ha='center', fontsize=5.5, color='#888888')
            else:
                ax.bar(x[i] + (k - 1) * w, v, w, color=['#2a78d6', '#eb6834', '#1baf7a'][k], label=task if i == 3 else None)
                if v == 0:
                    ax.text(x[i] + (k - 1) * w, 0.6, '0', ha='center', fontsize=5.5, color=['#2a78d6', '#eb6834', '#1baf7a'][k])
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=5.8)
    ax.set_ylabel('1日すべて帯の中の組（/40）')
    ax.set_ylim(0, 42)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_control.png')
    plt.close(fig)


def fig_band():
    """帯の幅ごとの30分コマの合格率（6 station、文書4.4節の値）。"""
    bins = ['<10', '10–15', '15–20', '20–30', '≥30']
    marl = {'AEMO': [68.3, 77.6, 89.1, 93.2, 99.2], 'ERCOT': [78.8, 79.4, 85.2, 95.7, 95.1], 'PJM': [52.4, 70.7, 82.2, 91.1, 100.0]}
    rule = {'AEMO': [99.0, 96.7, 96.1, 98.6, 99.4], 'ERCOT': [100.0, 97.8, 95.9, 99.0, 98.2], 'PJM': [93.7, 93.0, 96.7, 98.0, 100.0]}
    fig, ax = plt.subplots(figsize=(ONE, 1.8))
    for k, task in enumerate(marl):
        c = ['#2a78d6', '#eb6834', '#1baf7a'][k]
        ax.plot(bins, marl[task], 'o-', color=c, ms=3, lw=1, label=f'MARL {task}')
        ax.plot(bins, rule[task], 's--', color=c, ms=2.5, lw=0.8, alpha=0.6, label=f'ルール {task}')
    ax.set_xlabel('帯の幅（kW）')
    ax.set_ylabel('30分コマの合格率（%）')
    ax.set_ylim(45, 101)
    ax.legend(frameon=False, ncol=2, fontsize=5.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_band.png')
    plt.close(fig)


def fig_abg():
    """ABG と本線（6 station、40組）。"""
    out = {}
    for task, main, abg in [('AEMO', 'main6_aemo_v2/result_aemo_marl_4cmd_2ev.jsonl', 'abg6_aemo_v2/result_aemo_marl_4cmd_2ev.jsonl'),
                            ('ERCOT', 'main6_ercot_v2/result_ercot_marl_4cmd_2ev.jsonl', 'abg6_ercot_v2/result_ercot_marl_4cmd_2ev.jsonl')]:
        for name, f in (('本線', main), ('ABG', abg)):
            rs = rows(EVAL / f)
            err = np.concatenate([np.abs(r['traces']['residual_kw'])[np.asarray(r['traces']['tracking'], bool)] for r in rs])
            out[(task, name)] = (np.mean([r['step_pass_pct'] for r in rs]), np.mean([r['soc_hit_pct'] for r in rs]), err.mean())
    fig, axes = plt.subplots(1, 2, figsize=(ONE, 1.6))
    x = np.arange(2)
    for k, name in enumerate(('本線', 'ABG')):
        c = ['#2a78d6', '#eb6834'][k]
        axes[0].bar(x + (k - 0.5) * 0.35, [out[(t, name)][0] for t in ('AEMO', 'ERCOT')], 0.35, color=c, label=name)
        axes[0].bar(x + (k - 0.5) * 0.35, [out[(t, name)][1] for t in ('AEMO', 'ERCOT')], 0.35, color=c, alpha=0.35)
        axes[1].bar(x + (k - 0.5) * 0.35, [out[(t, name)][2] for t in ('AEMO', 'ERCOT')], 0.35, color=c)
    axes[0].set_ylabel('GLOBAL（濃）・LOCAL（淡）%')
    axes[0].set_ylim(40, 100)
    axes[1].set_ylabel('MAE（kW）')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(['AEMO', 'ERCOT'])
    axes[0].legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_abg.png')
    plt.close(fig)


def fig_selection():
    """選び方の表（6 station、AEMO は Windows、ERCOT は研究室の記録を写したもの）。上が AEMO、下が ERCOT。"""
    labels = {'main': '本線', 'cmd8': '', 'cmd16': '', 'cmd32': '指令8〜256本', 'cmd64': '', 'cmd256': '',
              'cnt1': 'EV1', 'cnt2': 'EV2', 'cnt4': 'EV4', 'cnt6': 'EV6', 'sel_low1medmax': '下1中最', 'sel_minmedmax': '最小中最',
              'sel_random0': '', 'sel_random1': '無作為（3回）', 'sel_random2': '',
              'acc95_low2max': '精度95', 'acc91_low2max': '精度91', 'acc85_low2max': '精度85',
              'acc95_random0': '95', 'acc91_random0': '91', 'acc85_random0': '85'}
    fig, axes = plt.subplots(2, 1, figsize=(ONE, 3.6), sharex=True)
    for ax, task in zip(axes, ('aemo', 'ercot')):
        files = {f.stem.replace(f'unseen_{task}_', ''): f for f in sorted(SEL.glob(f'unseen_{task}_*.jsonl'))}
        files.update({f.stem.replace(f'unseen_{task}_', ''): f for f in sorted((SEL / 'from_lab').glob(f'unseen_{task}_*.jsonl'))})
        for v, f in sorted(files.items()):
            rs = rows(f)
            if v not in labels:
                continue
            trials = [x for r in rs for x in r['rows']]
            if len(trials) != 1920:
                continue
            ok = 100 * sum(x['feasible'] is True for x in trials) / len(trials)
            cap = np.mean(list({(r['bank'], r['day']): r['offered_capacity_kw_block'] for r in rs}.values()))
            c = '#d03b3b' if v == 'main' else ('#2a78d6' if v.startswith('cmd') else '#eb6834' if v.startswith('cnt')
                                               else '#9c64c7' if v.startswith('acc') else '#888888')
            face = 'none' if v.startswith('acc') and v.endswith('random0') else c
            ax.scatter(cap / 1000, ok, s=10, color=c, facecolors=face, zorder=3)
            if labels[v]:
                offset = {'cmd32': (-6, -10), 'cnt6': (-4, -9), 'cnt1': (2, -8), 'sel_random1': (-2, -10), 'acc91_low2max': (4, 1)}.get(v, (2, 2))
                ax.annotate(labels[v], (cap / 1000, ok), fontsize=5, xytext=offset, textcoords='offset points')
        ax.set_ylabel('未見通過率（%）')
        ax.set_title(task.upper(), fontsize=6.5)
    axes[1].set_xlabel('入札量（MW・コマ／日）')
    fig.suptitle('紫は予測精度、白抜きは無作為に3通り（1,920組）', fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_selection.png')
    plt.close(fig)


def fig_bess():
    """蓄電池の出力と、月20日に換算した不適合の回数。"""
    fig, axes = plt.subplots(1, 2, figsize=(ONE, 2.2), sharey=True)
    for ax, prefix, title in ((axes[0], 'bess6', '6 station'), (axes[1], 'bess20', '20 station')):
        for task, c in (('aemo', '#2a78d6'), ('ercot', '#eb6834'), ('pjm', '#1baf7a')):
            for mode, ls in (('raw', '-'), ('force', '--')):
                f = EVAL / f'{prefix}_{task}_{mode}_v2.jsonl'
                if not f.exists():
                    continue
                by = defaultdict(list)
                for r in rows(f):
                    by[r['bess_kw']].append(r)
                xs = [kw for kw in sorted(by) if len(by[kw]) == 40]
                ys = [max(sum(r['post_blocks_failed'] for r in by[kw]) / 40 * 20, 0.3) for kw in xs]
                if xs:
                    ax.plot(np.array(xs) / 1000, ys, ls, color=c, lw=1, marker='o', ms=2,
                            label=f'{task.upper()}{"＋強制充電" if mode == "force" else ""}')
        ax.axhline(3, color='#d03b3b', lw=0.6, ls=':')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('蓄電池の出力（MW）')
    axes[0].set_ylabel('月あたりの不適合（回）')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.5, ncol=3, loc='lower center')
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / 'fig_bess.png')
    plt.close(fig)


def fig_placeholder(name: str, text: str, width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    placeholder(ax, text)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('#cccccc')
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    fig_final()
    fig_information()
    fig_control()
    fig_band()
    fig_abg()
    fig_selection()
    fig_bess()
    fig_placeholder('fig_causes.png', '5つの原因の影響度のまとめ（C0、未作成）', ONE, 1.8)
    fig_placeholder('fig_minbid.png', '最低入札量・余裕と成功率（C4、未測定）', ONE, 1.8)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
