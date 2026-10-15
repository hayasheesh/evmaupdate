"""固定40組の制御成績、ABG誤差分位点、基準値の段差別合格率を再集計する。"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SERIES=[('main6_aemo','aemo','marl','本線','#2a78d6'),
        ('abg6_aemo','aemo','marl','ABG','#eb6834'),
        ('rule6_aemo','aemo','rule','中央ルールベース','#1baf7a'),
        ('main6_ercot','ercot','marl','本線','#2a78d6'),
        ('abg6_ercot','ercot','marl','ABG','#eb6834'),
        ('rule6_ercot','ercot','rule','中央ルールベース','#1baf7a'),
        ('abg20_aemo','aemo','marl','ABG 20 station','#eb6834'),
        ('abg20_ercot','ercot','marl','ABG 20 station','#eb6834'),
        ('main20_aemo','aemo','marl','本線 20 station','#2a78d6'),
        ('main20_ercot','ercot','marl','本線 20 station','#2a78d6')]
SERIES += [('main6_pjm','pjm','marl','本線','#2a78d6'),
           ('rule6_pjm','pjm','rule','中央ルールベース','#1baf7a'),
           ('prior6_aemo','aemo','marl','先行方式の適用','#9c64c7')]
EDGES=[0,10,15,20,30,np.inf]
JUMPS=[0,10,25,50,100,np.inf]


def load(folder,task,mode):
    if folder.startswith('prior'):
        verification=HERE/(folder+'_v2')/'method_verification.json'
        if not verification.exists():return None
        proof=json.loads(verification.read_text(encoding='utf-8'))
        required={'upper_bid_policy','lower_ev_policies','dynamic_grouping','group_reference_generation',
                  'attention_value_network','group_value_mixing','all_actor_critic_updates','bid_control_feedback'}
        checked=proof.get('components',{})
        if not required.issubset(checked) or not all(checked[k].get('source_checked') and checked[k].get('implemented') and checked[k].get('tested') for k in required):
            return None
    f=HERE/(folder+'_v2')/f'result_{task}_{mode}_4cmd_2ev.jsonl'
    if not f.exists(): return None
    text=f.read_text(encoding='utf-8');lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    rows=[json.loads(x) for x in lines]
    if len(rows)!=40: return None
    keys=[(r['day'],r['command'],r['ev_seed']) for r in rows]
    if len(set(keys))!=40: raise RuntimeError('duplicate evaluation sets')
    return rows


def cells(rows,edges,field):
    result=[]
    for a,b in zip(edges[:-1],edges[1:]):
        selected=[(r,x) for r in rows for x in r['blocks'] if x.get(field) is not None and a<=x[field]<b]
        passed=[]
        for r,x in selected:
            start=x['k']*6;stop=start+6
            err=np.abs(r['traces']['residual_kw'][start:stop]);tol=np.asarray(r['traces']['tol_kw'][start:stop])
            mask=np.asarray(r['traces']['tracking'][start:stop],bool)
            passed.extend((err[mask]<=tol[mask]+1e-6).tolist())
        result.append({'from_kw':a,'to_kw':None if np.isinf(b) else b,'blocks':len(selected),
                       'pass_pct':float(100*np.mean([x['pass'] for _,x in selected])) if selected else None,
                       'global_tracking_pct':float(100*np.mean(passed)) if passed else None,
                       'assessed_5min_points':len(passed),
                       'first_step_mae_kw':float(np.mean([x['first_step_error_kw'] for _,x in selected])) if selected else None})
    return result


def main():
    plt.rcParams['font.family']=['Yu Gothic','Meiryo','sans-serif']
    report={}; loaded={}
    for folder,task,mode,label,color in SERIES:
        rows=load(folder,task,mode)
        if rows is None: continue
        loaded[folder]=rows
        errors=np.concatenate([np.abs(r['traces']['residual_kw'])[np.asarray(r['traces']['tracking'],bool)] for r in rows])
        failed=sum(r['blocks_failed'] for r in rows);total=sum(r['blocks_assessed'] for r in rows)
        item={'model':rows[0].get('model'),'episode':rows[0].get('episode'),'sets':40,
              'soc_hit_pct':float(np.mean([r['soc_hit_pct'] for r in rows])),
              'step_pass_pct':float(np.mean([r['step_pass_pct'] for r in rows])),
              'block_pass_pct':100*(1-failed/total),'failed_blocks':failed,'assessed_blocks':total,
              'mae_kw':float(np.mean(errors)), 'mean_error_to_award_pct':float(np.mean([r['mean_error_to_award_pct'] for r in rows])),
              'below_p90_sets':sum(r['step_pass_pct']<90 for r in rows),
              'monthly_failed_blocks_20days':failed/40*20,
              'global_tracking_pct':float(np.mean([r['step_pass_pct'] for r in rows])),
              'local_soc_pct':float(np.mean([r['soc_hit_pct'] for r in rows])),
              'global_local_sum':float(np.mean([r['step_pass_pct']+r['soc_hit_pct'] for r in rows])),
              'abs_error_quantiles_kw':{str(q):float(np.quantile(errors,q/100)) for q in (50,80,90,95,98,99)},
              'by_band':cells(rows,EDGES,'band_kw'),'by_baseline_step':cells(rows,JUMPS,'baseline_step_kw')}
        item['band_by_baseline_step']=[]
        for a,b in zip(EDGES[:-1],EDGES[1:]):
            subset=[dict(r,blocks=[x for x in r['blocks'] if a<=x['band_kw']<b]) for r in rows]
            item['band_by_baseline_step'].append({'band_from_kw':a,'band_to_kw':None if np.isinf(b) else b,
                                                  'steps':cells(subset,JUMPS,'baseline_step_kw')})
        report[folder]=item
    # AEMO本線の再評価が既存の40組と一致するかを確認する。
    if 'main6_aemo' in loaded:
        old_path=HERE.parent/'table3_resbid_6station_20261013'/'result_aemo_marl_4cmd_2ev.jsonl'
        old={(r['day'],r['command'],r['ev_seed']):r for r in map(json.loads,old_path.read_text(encoding='utf-8').splitlines())}
        diffs={key:max(abs(r[key]-old[(r['day'],r['command'],r['ev_seed'])][key]) for r in loaded['main6_aemo'])
               for key in ('soc_hit_pct','step_pass_pct','block_pass_pct')}
        report['aemo_difference_from_previous_evaluation']=diffs
        report['evaluation_normalization']='学習時の正規化を指令・入札先読み観測の生成前に読み込む。'
        repeated=load('main6_aemo_repeat1','aemo','marl')
        if repeated:
            reference={(r['day'],r['command'],r['ev_seed']):r for r in loaded['main6_aemo']}
            repeat_diffs={key:max(abs(r[key]-reference[(r['day'],r['command'],r['ev_seed'])][key]) for r in repeated)
                          for key in ('soc_hit_pct','step_pass_pct','block_pass_pct','mae_kw')}
            report['aemo_worker_count_reproduction_max_abs_diff']=repeat_diffs
            if max(repeat_diffs.values())>1e-6:raise RuntimeError(f'corrected evaluation depends on worker count: {repeat_diffs}')
    # 同じ市場・規模では、指令・帯・基準値・EVの組を一致させる。
    for task in ('aemo','ercot','pjm'):
        group=[rows for folder,rows in loaded.items() if '6_'+task in folder]
        if len(group)<2:continue
        reference={(r['day'],r['command'],r['ev_seed']):r for r in group[0]}
        for rows in group[1:]:
            for r in rows:
                base=reference[(r['day'],r['command'],r['ev_seed'])]
                if r['departed_evs']!=base['departed_evs'] or any(r['traces'][k]!=base['traces'][k] for k in ('target_kw','tol_kw','baseline_kw','up_kw','down_kw')):
                    raise RuntimeError('controller comparison uses different physical inputs')
        report[f'{task}_paired_inputs_verified']=True
    summary=HERE/'control_summary.json';tmp=summary.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(summary)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,task in zip(axes,('aemo','ercot')):
        for folder,t,mode,label,color in SERIES:
            if t!=task or folder not in report or '20_' in folder or folder.startswith(('abg','prior')): continue
            vals=[x['global_tracking_pct'] if x['blocks'] else np.nan for x in report[folder]['by_band']]
            ax.plot(range(5),vals,'o-',label=label,color=color,lw=2)
        ax.set_xticks(range(5),['<10','10–15','15–20','20–30','≥30']);ax.set_xlabel('帯の幅（kW）')
        ax.set_title(task.upper(),loc='left');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.25)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('GLOBAL：5分ごとの追従率（%）')
    if axes[1].get_legend_handles_labels()[0]:axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('本線と中央ルールベース：帯の幅による追従率の違い',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(HERE/'fig_abg_band.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharey=True)
    for ax,task in zip(axes,('aemo','ercot')):
        for folder,t,mode,label,color in SERIES:
            if t!=task or folder not in report or not folder.startswith('main6_'): continue
            vals=report[folder]['by_baseline_step']
            shown=[x for x,v in enumerate(vals) if v['blocks']]
            ax.scatter(shown,[vals[x]['global_tracking_pct'] for x in shown],label=label,color=color,s=45)
            if label=='本線':
                for x,v in enumerate(vals):
                    if v['blocks']:ax.annotate(f"n={v['blocks']}コマ",(x,v['global_tracking_pct']),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(5),['<10','10–25','25–50','50–100','≥100']);ax.set_xlabel('隣接する参加コマの基準値の段差（kW）')
        ax.set_title(task.upper(),loc='left');ax.set_ylim(0,108);ax.grid(axis='y',alpha=.25)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('GLOBAL：5分ごとの追従率（%）')
    if axes[1].get_legend_handles_labels()[0]:axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('診断用：基準値の段差とGLOBAL（本線、同じ40組）',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(HERE/'fig_baseline_step.png',dpi=160);plt.close(fig)
    comparison_tasks=[task for task in ('aemo','ercot','pjm') if f'main6_{task}' in report]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    x=np.arange(len(comparison_tasks))
    for shift,method,label,color in [(-.18,'main','本線','#2a78d6'),(.18,'rule','中央ルールベース','#1baf7a')]:
        vals=[report.get(f'{method}6_{task}',{}).get('global_tracking_pct',np.nan) for task in comparison_tasks]
        axes[0].bar(x+shift,vals,.34,color=color,label=label)
        for xx,v in zip(x+shift,vals):
            if np.isfinite(v):axes[0].text(xx,v+2,f'{v:.1f}',ha='center',fontsize=9)
    main_local=[report.get(f'main6_{task}',{}).get('local_soc_pct',np.nan) for task in comparison_tasks]
    rule_local=[report.get(f'rule6_{task}',{}).get('local_soc_pct',np.nan) for task in comparison_tasks]
    constant_rule=all(np.isfinite(v) and abs(v-100)<1e-8 for v in rule_local)
    if constant_rule:
        axes[1].bar(x,main_local,.5,color='#2a78d6')
        for xx,v in zip(x,main_local):
            if np.isfinite(v):axes[1].text(xx,v+2,f'{v:.1f}',ha='center',fontsize=9)
    else:
        for shift,vals,color in [(-.18,main_local,'#2a78d6'),(.18,rule_local,'#1baf7a')]:
            axes[1].bar(x+shift,vals,.34,color=color)
    axes[1].axhline(100,color='#777',lw=1,ls='--')
    for ax in axes:
        ax.set_xticks(x,[task.upper() for task in comparison_tasks]);ax.set_ylim(0,110);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('GLOBAL：5分ごとの追従率（%）');axes[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.10),ncol=2,fontsize=9)
    axes[1].set_ylabel('LOCAL：SoC達成率（%）');axes[1].set_title('本線の充電達成率' if constant_rule else '充電達成率',fontsize=11)
    fig.suptitle('固定入札での制御性能：6 station、40組、蓄電池なし',fontsize=12)
    if constant_rule:fig.text(.5,.015,'中央ルールベースのLOCALは表示した市場で100%。和と詳細値は表に掲載。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(HERE/'fig_control_global_local.png',dpi=160);plt.close(fig)
    pairs=[(n,task) for n in (6,20) for task in ('aemo','ercot')
           if f'main{n}_{task}' in report and f'abg{n}_{task}' in report]
    if pairs:
        fig,axes=plt.subplots(len(pairs),2,figsize=(8.5,3.0*len(pairs)),squeeze=False)
        for i,(n,task) in enumerate(pairs):
            base=report[f'main{n}_{task}'];abg=report[f'abg{n}_{task}']
            for j,(field,label) in enumerate([('global_tracking_pct','GLOBAL：5分ごとの追従率（%）'),
                                              ('mae_kw','MAE（kW）')]):
                ax=axes[i,j];vals=[base[field],abg[field]]
                ax.bar(range(2),vals,color=['#2a78d6','#eb6834'],width=.6)
                top=108 if j==0 else max(vals)*1.30
                for x,v in enumerate(vals):ax.text(x,v+top*.02,f'{v:.2f}',ha='center',fontsize=10)
                ax.set_xticks(range(2),[f'本線 ep{base["episode"]}',f'ABG ep{abg["episode"]}'])
                ax.set_ylabel(label);ax.set_ylim(0,top);ax.set_title(f'{n} station {task.upper()}',loc='left')
                ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        fig.suptitle('本線とABGの追従性能：同じ入札・指令・EV、40組、蓄電池なし',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(HERE/'fig_abg_tracking_mae.png',dpi=160);plt.close(fig)
    if all(key in report for key in ('main6_aemo','prior6_aemo','rule6_aemo')):
        fig,axes=plt.subplots(1,2,figsize=(9.5,4.5))
        keys=['main6_aemo','prior6_aemo','rule6_aemo']
        labels=['本線','先行方式の適用','中央ルールベース']
        for ax,field,ylabel in zip(axes,('global_tracking_pct','local_soc_pct'),
                                  ('GLOBAL：5分ごとの追従率（%）','LOCAL：SoC達成率（%）')):
            values=[report[key][field] for key in keys]
            ax.bar(range(3),values,color=['#2a78d6','#9c64c7','#1baf7a'],width=.6)
            for i,value in enumerate(values):ax.text(i,value+2,f'{value:.1f}',ha='center')
            ax.set_xticks(range(3),labels,fontsize=9);ax.set_ylim(0,110);ax.set_ylabel(ylabel)
            ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        fig.suptitle('共通入札での先行方式との比較：6 station AEMO、同じ40組',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(HERE/'fig_prior_marl_aemo.png',dpi=160);plt.close(fig)
    print(json.dumps({k:{f:v[f] for f in ('soc_hit_pct','step_pass_pct','block_pass_pct','mae_kw')} for k,v in report.items() if isinstance(v,dict) and 'sets' in v},ensure_ascii=False))


if __name__=='__main__':main()
