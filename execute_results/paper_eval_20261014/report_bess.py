"""BESS出力別の不適合とEV充電達成率を並べて集計する。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
plt.rcParams['font.family']=['Yu Gothic','Meiryo','sans-serif']
report={}
for task in ('aemo','ercot','pjm'):
    for mode in ('raw','force'):
        path=HERE/f'bess6_{task}_{mode}_v2.jsonl'
        if not path.exists():continue
        text=path.read_text(encoding='utf-8');lines=text.splitlines()
        if text and not text.endswith('\n'):lines=lines[:-1]
        rows=[json.loads(x) for x in lines]
        points=[]
        for size in sorted({r['bess_kw'] for r in rows}):
            subset=[r for r in rows if r['bess_kw']==size]
            if len(subset)!=40:continue
            if len({(r['day'],r['command'],r['ev_seed']) for r in subset})!=40:raise RuntimeError('BESS sets duplicated')
            failed=sum(r['post_blocks_failed'] for r in subset);total=sum(r['post_blocks_assessed'] for r in subset)
            points.append({'bess_kw':size,'bess_kwh':subset[0]['bess_kwh'],'sets':40,'failed_blocks':failed,
                           'block_pass_pct':100*(1-failed/total),'monthly_expected_failed_blocks':failed/40*20,
                           'soc_hit_pct':float(np.mean([r['soc_hit_pct'] for r in subset])),
                           'inference_ms_p95_mean':float(np.mean([r['inference_ms_p95'] for r in subset]))})
        report[f'{task}_{mode}']=points
(HERE/'bess_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if any(report.values()):
    plot_tasks=[task for task in ('aemo','ercot','pjm') if any(report.get(f'{task}_{mode}') for mode in ('raw','force'))]
    fig,axes=plt.subplots(1,len(plot_tasks),figsize=(5*len(plot_tasks),4.5),sharey=True,squeeze=False)
    axes=axes[0]
    for j,task in enumerate(plot_tasks):
        for mode,label,color in [('raw','行動器＋BESS','#2a78d6'),('force','行動器＋強制充電＋BESS','#eb6834')]:
            rows=report.get(f'{task}_{mode}',[])
            if not rows:continue
            x=[r['bess_kw'] for r in rows]
            axes[j].plot(x,[r['monthly_expected_failed_blocks'] for r in rows],'o-',label=label,color=color)
        axes[j].set_title(task.upper(),loc='left');axes[j].set_yscale('symlog',linthresh=1)
        axes[j].axhline(3,ls='--',color='#777',label='月3回')
        axes[j].set_xlabel('BESS出力（kW）')
        axes[j].grid(axis='y',alpha=.25);axes[j].spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('月20日に換算した不適合（回）')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('BESS出力と市場の不適合：各点40組、充電達成率は表に掲載',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(HERE/'fig_bess_charging.png',dpi=160);plt.close(fig)
print(json.dumps({k:len(v) for k,v in report.items()}))
