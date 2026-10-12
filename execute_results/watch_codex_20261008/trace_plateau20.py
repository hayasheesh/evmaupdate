"""同一指令・EV実現で行動器と診断用介入を比較する。学習は行わない。"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PREPARED=None
ACTORS=None
SCREEN=None
DEST=None
BEST={'aemo':1380,'ercot':1340}
LATE={'aemo':1620,'ercot':1580}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def setup(market,prepared,dest):
    global PREPARED,ACTORS,SCREEN,DEST
    # Reuse environment preparation helpers, not persisted result rows.
    spec=importlib.util.spec_from_file_location('screen',ROOT/'execute_results/watch_codex_20261008/trace_screen_helpers.py')
    SCREEN=importlib.util.module_from_spec(spec);spec.loader.exec_module(SCREEN)
    SCREEN.ROOT=ROOT;SCREEN.configure(market)
    sys.path.insert(0,str(ROOT))
    import torch
    from environment.normalize import load_observation_normalization_for_archive
    torch.set_num_threads(1)
    load_observation_normalization_for_archive(SCREEN.roots(market)[2])
    PREPARED=prepared;SCREEN.PREPARED=prepared;SCREEN.CONSTANTS={};SCREEN.MODE='normal'
    ACTORS={k:SCREEN.load_actors(market,ep) for k,ep in [('best',BEST[market]),('late',LATE[market])]}
    DEST=Path(dest)

def rollout(case):
    import numpy as np
    import torch
    from EnvConfig import POWER_TO_ENERGY
    from environment.normalize import normalize_observation
    mode=case['mode'];actors=ACTORS['best' if mode=='best' else 'late']
    if mode.startswith('swap_'):
        stations={int(x)-1 for x in mode.split('_')[1:]}
        actors=[ACTORS['best'][i] if i in stations else ACTORS['late'][i] for i in range(20)]
    env,p=SCREEN.make_env(case)
    frozen={'events':env._arrival_events,'soc':env.soc.cpu().tolist(),'target':env.target.cpu().tolist(),
            'depart':env.depart.cpu().tolist(),'cap':env.ev_capacity_kwh.cpu().tolist(),'power':env.ev_max_power_kw.cpu().tolist()}
    evhash=hashlib.sha256(json.dumps(frozen,sort_keys=True,default=str).encode()).hexdigest()
    erows=[];steps=[]
    with torch.no_grad():
        for _ in range(288):
            obs=normalize_observation(env.begin_step())
            action=torch.stack([actors[i](obs[i]) for i in range(20)])
            before=[];lower=[];upper=[];floors=[];nominal=[]
            for st in range(20):
                slots=env._get_sorted_active_evs(st)
                for j,slot in enumerate(slots.tolist()):
                    soc=float(env.soc[st,slot]);target=float(env.target[st,slot]);cap=float(env.ev_capacity_kwh[st,slot]);power=float(env.ev_max_power_kw[st,slot]);remain=int(env.depart[st,slot])-env.step_count+1
                    lo=max(-power,-soc*cap/100/POWER_TO_ENERGY);hi=min(power,(100-soc)*cap/100/POWER_TO_ENERGY)
                    floor=max(lo,(target-soc)*cap/100/POWER_TO_ENERGY-(remain-1)*power)
                    a=float(action[st,j]);want=max(-1,min(1,a))*power
                    before.append((st,j,slot,int(env.ev_ids[st,slot]),float(env.arrival_step[st,slot]),remain,soc,target,cap,power,a))
                    lower.append(lo);upper.append(hi);floors.append(floor);nominal.append(want)
            low=np.array(lower);high=np.array(upper);floor=np.array(floors);nom=np.array(nominal);physical=np.clip(nom,low,high)
            req=float(env.current_net_demand);band=float(env.tol_narrow_metrics);enabled=bool(env.current_tracking_enabled)
            feasible=low.sum()<=req+band+1e-5 and high.sum()>=req-band-1e-5
            safe=(floor<=high+1e-5).all() and floor.sum()<=req+band+1e-5 and high.sum()>=req-band-1e-5
            if mode in ('deadline_floor','deadline_floor_buffer','safe_projection'):
                safe_low=floor+(0 if mode=='deadline_floor' else 1e-4/POWER_TO_ENERGY)
                physical=np.clip(physical,np.minimum(safe_low,high),high)
            if mode=='cancel_counterflow':
                for st in range(20):
                    ii=np.array([i for i,b in enumerate(before) if b[0]==st],dtype=int)
                    if not len(ii):continue
                    v=physical[ii].copy();pos=np.maximum(v,0).sum();neg=np.maximum(-v,0).sum()
                    if pos>=neg:
                        v=np.maximum(v,0)*(pos-neg)/max(pos,1e-9)
                    else:v=np.minimum(v,0)*(neg-pos)/max(neg,1e-9)
                    physical[ii]=v
            if mode in ('instant_projection','safe_projection') and enabled:
                projection_low=low if mode=='instant_projection' else np.minimum(safe_low,high)
                wanted=np.clip(req,projection_low.sum(),high.sum());change=wanted-physical.sum()
                head=(high-physical) if change>=0 else (physical-projection_low)
                if head.sum()>1e-8:physical=physical+np.sign(change)*min(abs(change),head.sum())*head/head.sum()
            if mode not in ('best','late') and not mode.startswith('swap_'):
                for i,b in enumerate(before):action[b[0],b[1]]=float(physical[i]/b[9])
            _,_,_,_,info=env.apply_action(action,return_observation=False,include_active_evs=False,include_ev_physics=False,include_reward_breakdown=False)
            executed=info['actual_ev_power_kw'].cpu().numpy()
            charge=0.;discharge=0.;urgent_discharge=0.;debt_count=0;debt_kwh=0.;depart_miss=0;depart_count=0;clipped_kwh=0
            for i,b in enumerate(before):
                st,j,slot,eid,arrival,remain,soc,target,cap,power,a=b
                actual=float(executed[st,j]);after=min(100,max(0,soc+actual*POWER_TO_ENERGY*100/cap))
                margin=(remain-1)*power*POWER_TO_ENERGY-max((target-after)*cap/100,0)
                charge+=max(actual,0);discharge+=max(-actual,0)
                if remain<=12 and soc<target-1e-3:urgent_discharge+=max(-actual,0)
                if margin<-1e-3:debt_count+=1;debt_kwh+=-margin
                if remain==1:depart_count+=1;depart_miss+=int(after<target-1e-3)
                clipped_kwh+=abs(nom[i]-actual)*POWER_TO_ENERGY if mode in ('best','late') else 0
                erows.append([env.step_count,st,eid,arrival,remain,soc,after,target,cap,power,a,nom[i],actual,low[i],high[i],floor[i],margin])
            residual=float(info['raw_actor_residual_kw'])
            steps.append([env.step_count,enabled,req,band,residual,feasible,safe,len(before),charge,discharge,urgent_discharge,debt_count,debt_kwh,depart_count,depart_miss,clipped_kwh])
    e=np.asarray(erows,dtype=np.float32);z=np.asarray(steps,dtype=np.float64);active=z[:,1]>0;bad=active&(np.abs(z[:,4])>z[:,3]+1e-5)
    m=env.get_metrics();passed=np.abs(z[:,4])<=z[:,3];en=active.reshape(48,6);blocks=en.any(1)
    row={'id':case['id'],'mode':mode,'market':case['market'],'date':p['date'],'command':p['source'],'ev_seed':case['ev_seed'],'ev_hash':evhash,
         'tracking':float(m['tracking_success_rate']),'soc':100-float(m['soc_miss_rate']),'mae':float(m['raw_actor_mae_kw']),
         'block_pass':float(100*(passed.reshape(48,6)|~en).all(1)[blocks].mean()),
         'failed_steps':int(bad.sum()),'failed_with_instant_headroom':int((bad&(z[:,5]>0)).sum()),
         'failed_without_instant_headroom':int((bad&(z[:,5]==0)).sum()),'failed_with_deadline_safe_headroom':int((bad&(z[:,6]>0)).sum()),
         'charge_kwh':float(z[:,8].sum()*POWER_TO_ENERGY),'discharge_kwh':float(z[:,9].sum()*POWER_TO_ENERGY),
         'simultaneous_charge_discharge_kwh':float(np.minimum(z[:,8],z[:,9]).sum()*POWER_TO_ENERGY),
         'urgent_discharge_kwh':float(z[:,10].sum()*POWER_TO_ENERGY),'unrecoverable_ev_steps':int(z[:,11].sum()),
         'soc_clipped_kwh':float(z[:,15].sum()),'depart_count':int(z[:,13].sum()),'depart_miss':int(z[:,14].sum())}
    row['sum']=row['tracking']+row['soc']
    file=DEST/(case['mode']+'_'+case['id'])
    np.savez_compressed(str(file)+'.npz',ev=e,step=z)
    Path(str(file)+'.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('market',choices=['aemo','ercot']);ap.add_argument('--prepared',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--modes',nargs='+',default=['best','late','deadline_floor','instant_projection'])
    ap.add_argument('--reference-root');ap.add_argument('--all-cases',action='store_true');a=ap.parse_args()
    class NumpyCompatibleUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            # The trusted Windows cache uses NumPy 2's renamed module path.
            if module.startswith('numpy._core'):
                module=module.replace('numpy._core','numpy.core',1)
            return super().find_class(module,name)
    with Path(a.prepared).open('rb') as f:prepared,cases=NumpyCompatibleUnpickler(f).load()
    selected=cases if a.all_cases else [c for c in cases if '_c0_' in c['id'] and c['id'].endswith(('_e0','_e1'))]
    if not a.all_cases:assert len(selected)==10
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    meta={'market':a.market,'cases':len(selected),'prepared_sha256':digest(a.prepared),'script_sha256':digest(__file__),
          'best_episode':BEST[a.market],'late_episode':LATE[a.market],
          'scope':'paired diagnostic cases in supplied prepared data; inspect case dates/commands/seeds for coverage; no unseen-generalization claim',
          'instant_feasibility':'SoC 0..100 and rated power only; ignores future instructions and departure targets',
          'deadline_safe_feasibility':'one-step lower bound permits target by rated charging thereafter; not a full future LP',
          'ev_columns':['step','station','id','arrival_step','remaining_steps','soc_before','soc_after','target','capacity_kwh','rated_kw','policy_action','nominal_kw','executed_kw','instant_min_kw','instant_max_kw','deadline_min_kw','recoverable_margin_after_kwh'],
          'step_columns':['step','tracking_enabled','request_kw','band_kw','residual_kw','instant_feasible','deadline_safe_feasible','EV_count','charge_kw','discharge_kw','urgent_discharge_kw','unrecoverable_EV_count','unrecoverable_debt_kwh','depart_count','depart_miss','soc_clipped_kwh']}
    names={'aemo':'allin_aemoplan_AB_20station_20261007_042318','ercot':'allin_ercotplan_AB_20station_20261007_042436'}
    meta['actor_sha256']={str(ep):[digest(ROOT/'archive'/names[a.market]/f'results/TEST{ep}/actor_{i}_ep{ep}.pth') for i in range(20)] for ep in (BEST[a.market],LATE[a.market])}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    jobs=[dict(c,mode=mode,market=a.market) for mode in a.modes for c in selected]
    rows=[]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=setup,initargs=(a.market,prepared,str(out))) as pool:
        futures=[pool.submit(rollout,c) for c in jobs]
        for f in as_completed(futures):
            rows.append(f.result());print(a.market,'completed',len(rows),'/',len(jobs),flush=True)
    if a.reference_root:
        previous=json.loads((Path(a.reference_root)/'summary.json').read_text())
        assert previous['metadata']['prepared_sha256']==meta['prepared_sha256']
        assert previous['metadata']['actor_sha256']==meta['actor_sha256']
        ref={r['id']:r for r in previous['rows'] if r['mode']=='late'}
    else:ref={r['id']:r for r in rows if r['mode']=='late'}
    for r in rows:assert r['ev_hash']==ref[r['id']]['ev_hash'] and r['command']==ref[r['id']]['command'] and r['depart_count']==ref[r['id']]['depart_count']
    import numpy as np
    summary={mode:{k:float(np.mean([r[k] for r in rows if r['mode']==mode])) for k in rows[0] if isinstance(rows[0][k],(float,int)) and k!='ev_seed'} for mode in a.modes}
    (out/'summary.json').write_text(json.dumps({'metadata':meta,'summary':summary,'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
