"""同じ40組の入札量を全情報条件と比較し、損と成否を図にする。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ER = HERE.parent
EXPECTED = {(d,c,e) for d in range(5) for c in range(4) for e in range(2)}
SELS = [('low2max','接続の少ない2通り＋最多','#2a78d6'),
        ('minmedmax','EV総数の最小・中央・最大','#eb6834'),
        ('random3','無作為3通り','#1baf7a')]


def read(path):
    text=path.read_text(encoding='utf-8')
    lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    rows = [json.loads(x) for x in lines]
    keys = [(r['day'],r['command'],r['ev_seed']) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= EXPECTED:
        raise RuntimeError(f'duplicated or unexpected set: {path}')
    return rows


def summary(rows, top):
    if {(r['day'],r['command'],r['ev_seed']) for r in rows} != EXPECTED:
        return {'complete':False,'sets':len(rows)}
    cap = np.mean([r['capacity_kw_block'] for r in rows])
    return {'complete':True,'sets':40,'capacity_kw_block_day':float(cap),
            'oracle_ratio_pct':float(100*cap/top),'loss_pct':float(100*(1-cap/top)),
            'feasible':sum(r['feasible'] is True for r in rows),
            'infeasible':sum(r['feasible'] is False for r in rows),
            'unknown':sum(r['feasible'] is None for r in rows)}


def main():
    plt.rcParams['font.family'] = ['Yu Gothic','Meiryo','sans-serif']
    report = {'definition':'loss = 100 * (1 - mean_capacity / mean_oracle_capacity)',
              'capacity_unit':'kW・コマ／日（上げと下げの合計）','sweep':{},'information':{},
              'calibration':json.loads((ER/'bid_loss_6station_20261013'/'calib.json').read_text(encoding='utf-8'))}
    for task in ('aemo','ercot'):
        for n in (6,20):
            folder = ER/f'info_bid_{n}station_202610{12 if n==6 else 13}'
            oracle = read(folder/f'result_{task}_oracle.jsonl')
            if len(oracle) != 40:
                raise RuntimeError('oracle must have all 40 sets')
            top = float(np.mean([r['capacity_kw_block'] for r in oracle]))
            for mode in ('oracle','cmd','ev'):
                report['information'][f'{n}_{task}_{mode}'] = summary(read(folder/f'result_{task}_{mode}.jsonl'),top)
            report['information'][f'{n}_{task}_main'] = {
                'capacity_kw_block_day':float(np.mean([r['main_capacity_kw_block'] for r in oracle])),
                'oracle_ratio_pct':float(100*np.mean([r['main_capacity_kw_block'] for r in oracle])/top)}
        top = report['information'][f'6_{task}_oracle']['capacity_kw_block_day']
        for cmd in ('known','unknown'):
            for acc in ('85','91','95'):
                for sel,_,_ in SELS:
                    p = ER/'bid_loss_6station_20261013'/f'result_{task}_{cmd}_{acc}_{sel}.jsonl'
                    if p.exists():
                        report['sweep'][f'{task}_{cmd}_{acc}_{sel}'] = summary(read(p),top)
    (HERE/'bid_loss_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    ymax=max(v['loss_pct'] for v in report['sweep'].values() if v.get('complete'))+8
    for i,task in enumerate(('aemo','ercot')):
        for j,cmd in enumerate(('known','unknown')):
            ax=axes[i,j]
            for series_i,(sel,label,color) in enumerate(SELS):
                vals=[report['sweep'].get(f'{task}_{cmd}_{a}_{sel}',{}) for a in ('85','91','95')]
                ys=[v.get('loss_pct',np.nan) if v.get('complete') else np.nan for v in vals]
                ax.plot([85,91,95],ys,'o-',label=label,color=color,lw=2)
            ax.set_title(task.upper()+'／指令'+('既知' if cmd=='known' else '未知'),loc='left')
            ax.set_xticks([85,91,95]); ax.set_ylim(0,ymax)
            ax.grid(axis='y',alpha=.25); ax.spines[['top','right']].set_visible(False)
            if i==1: ax.set_xlabel('EV予測の精度（100−接続台数のWAPE、%）')
            if j==0: ax.set_ylabel('両方全知の入札量からの減少（%）')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('6 station：予測精度・EVの選び方による入札量の減少',fontsize=12)
    fig.text(.5,.06,'各点40組。未完了条件は集計後に追加。未見成立率は表に掲載。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.93))
    fig.savefig(HERE/'fig_bid_loss_accuracy.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    modes=[('ev','EVだけ既知'),('cmd','指令だけ既知'),('main','両方未知')]
    for ax,task in zip(axes,('aemo','ercot')):
        x=np.arange(3)
        for shift,n,color in [(-.18,6,'#2a78d6'),(.18,20,'#eb6834')]:
            vals=[report['information'][f'{n}_{task}_{m}']['oracle_ratio_pct'] for m,_ in modes]
            ax.bar(x+shift,vals,.34,label=f'{n} station',color=color)
            for xx,y in zip(x+shift,vals): ax.text(xx,y+1,f'{y:.1f}',ha='center',fontsize=8)
        ax.set_xticks(x,[label for _,label in modes]); ax.set_ylim(0,108)
        ax.axhline(100,color='#777',lw=1,ls='--')
        ax.set_title(task.upper(),loc='left'); ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('両方全知の入札量を100%とした比（%）')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('情報条件による入札量の比較：同じ40組',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,.94)); fig.savefig(HERE/'fig_information_capacity.png',dpi=160); plt.close(fig)
    print(json.dumps({'complete_conditions':sum(v['complete'] for v in report['sweep'].values()),'total_conditions':36},ensure_ascii=False))


if __name__=='__main__': main()
