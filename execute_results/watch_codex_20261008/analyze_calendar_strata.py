"""帯幅・依頼/定格・時間帯をそろえた休日別の記述集計。因果推定ではない。"""
import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('calendar_only',ROOT/'environment/calendars.py')
cal=importlib.util.module_from_spec(spec);spec.loader.exec_module(cal)
result={}
for market in ['aemo','ercot']:
    folder=BASE/f'calendar20_{market}_20261008';summary=json.loads((folder/'summary.json').read_text());cells={};totals=np.zeros((2,2));bands=[[],[]]
    for row in summary['rows']:
        if row['mode']!='late':continue
        group=int(cal.day_class(row['date'],'JP')=='holiday')
        a=np.load(folder/(row['mode']+'_'+row['id']+'.npz'));e=a['ev'];z=a['step']
        rated=np.bincount(e[:,0].astype(int)-1,weights=e[:,9],minlength=288)
        for i,x in enumerate(z):
            if not x[1]:continue
            fail=int(abs(x[4])>x[3]);ratio=abs(x[2])/max(rated[i],1e-9)
            key=(int(np.digitize(x[3],[100,200])),int(np.digitize(ratio,[.25,.5,.75])),int((x[0]-1)//72))
            cell=cells.setdefault(key,np.zeros((2,2)));cell[group]+=[1,fail];totals[group]+=[1,fail];bands[group].append(x[3])
    shared=[v for v in cells.values() if np.all(v[:,0]>=30)]
    weights=np.array([v[:,0].sum() for v in shared]);weights=weights/weights.sum()
    rate=np.sum(np.stack([v[:,1]/v[:,0] for v in shared])*weights[:,None],axis=0)
    result[market]={'definitions':'weekday=0, weekend/public holiday=1. Cells: band<100/100..200/>=200kW, absrequest/connectedrated<.25/.25..5/.5..75/>=.75, 6-hour intervals. Require >=30 observed assessed steps in each group per shared cell; descriptive standardization, not causal inference.',
                    'unadjusted_outside_pct':(100*totals[:,1]/totals[:,0]).tolist(),'common_stratum_standardized_outside_pct':(100*rate).tolist(),
                    'covered_steps_by_group':np.sum([v[:,0] for v in shared],axis=0).astype(int).tolist(),'total_steps_by_group':totals[:,0].astype(int).tolist(),'shared_cells':len(shared),'band_mean_by_group':[float(np.mean(v)) for v in bands]}
out=BASE/'calendar_strata_analysis.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
