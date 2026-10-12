"""保存行動器の観測群を固定して同一条件で比較するCPU解析。"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
P = argparse.ArgumentParser()
P.add_argument('market', choices=['aemo', 'ercot'])
P.add_argument('--workers', type=int, default=12)
P.add_argument('--commands', type=int, default=8)
P.add_argument('--ev-seeds', type=int, default=3)
P.add_argument('--episode', type=int, default=1100)
P.add_argument('--smoke', action='store_true')
ARGS = None
ACTORS = None
PREPARED = None
CONSTANTS = None
GROUPS = None
MODE = 'normal'

def configure(market):
    for k in list(os.environ):
        if k.startswith('EVMA_'): del os.environ[k]
    signal = market+'_plan_deviation'
    os.environ.update(EVMA_NUM_STATIONS='20', EVMA_MARL_ALGORITHM='hybrid',
                      EVMA_ACTIVATION_SIGNAL_SET=signal,
                      EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION='low_connection_2_max_count',
                      CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')

def roots(market):
    signal=market+'_plan_deviation'
    base=ROOT/'execute_results/bid_banks'
    train=base/f'low2max_margin80_train_25_20station_128cmd_3ev_{signal}'
    valid=base/f'low2max_margin80_validation_5_20station_128cmd_3ev_{signal}'
    run=ROOT/'archive'/({'aemo':'allin_aemoplan_AB_20station_20261007_042318',
                        'ercot':'allin_ercotplan_AB_20station_20261007_042436'}[market])
    return train,valid,run

def group_indices():
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES as tail, EV_FEATURE_NAMES as ev
    offset=10*len(ev); g={}
    for j,name in enumerate(ev):
        if name!='presence':g['EV_'+name]=list(range(j,offset,len(ev)))
    for j,name in enumerate(tail):
        if 'lookahead' not in name:g[name]=[offset+j]
    g['active_EV_count']=[]
    for field in ('baseline','up','down'):
        g['bid_'+field]=[offset+j for j,n in enumerate(tail) if n.startswith('bid_'+field+'_lookahead_')]
    g['bid_current']=[offset+j for j,n in enumerate(tail) if 'lookahead_0' in n]
    for low,high in ((1,5),(6,11),(12,23),(1,23),(0,23)):
        g[f'bid_offsets_{low}_{high}']=[offset+j for j,n in enumerate(tail) if 'lookahead_' in n and low<=int(n.rsplit('_',1)[1])<=high]
    g['SOC_and_needed']=sorted(set(g['EV_soc']+g['EV_needed_soc']))
    g['band_and_current_bid']=sorted(set(g['tracking_band_kw']+g['bid_current']))
    g['scale_and_baseline_bid']=sorted(set(g['instruction_scale_kw']+g['bid_baseline']))
    g['holiday_and_clock']=sorted(set(g['service_day_holiday']+g['step']))
    return g

def load_actors(market,episode):
    import torch
    from training.Agent.actor import Actor
    from environment.observation_config import local_obs_dim
    _,_,run=roots(market); models=[]
    for i in range(20):
        actor=Actor(local_obs_dim(10),10)
        file=run/f'results/TEST{episode}/actor_{i}_ep{episode}.pth'
        actor.load_state_dict(torch.load(file,map_location='cpu',weights_only=True))
        actor.eval()
        def hook(module,args):
            if MODE in CONSTANTS.get('count_modes',['active_EV_count']):
                x=args[0].clone();x[...,-1]=float(CONSTANTS['active_count']);return (x,)
        actor.ev_action_head.register_forward_pre_hook(hook)
        models.append(actor)
    return models

def make_env(case):
    import numpy as np
    from tools.evaluator import set_env_seed
    from environment.EVEnv import EVEnv
    from environment.normalize import use_instruction_scale
    p=PREPARED[case['payload']]
    set_env_seed(case['ev_seed']);use_instruction_scale(p['arr'].get('instruction_scale_kw',1.0))
    env=EVEnv();env.use_central_ev_residual_allocator=False;env.use_residual_bess=False
    env.reset(net_demand_series=p['target'],tol_narrow_series=p['tol'],tracking_enabled_series=p['tracking'],
              market_context_series=p['arr'].get('market_context_series'),
              arrival_probabilities_by_station=p['arr'].get('arrival_probabilities_by_station'),
              service_date=p['date'],baseline_series=p['arr'].get('baseline_series'))
    return env,p

def rollout(case,collect=False):
    global MODE
    import numpy as np
    import torch
    from environment.normalize import normalize_observation
    MODE=case['mode'];env,p=make_env(case)
    frozen={'arrivals':env._arrival_events,'soc':env.soc.cpu().tolist(),'target':env.target.cpu().tolist(),
            'depart':env.depart.cpu().tolist(),'cap':env.ev_capacity_kwh.cpu().tolist(),'power':env.ev_max_power_kw.cpu().tolist()}
    ev_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True,default=str).encode()).hexdigest()
    collected=[];success=[];enabled=[]
    with torch.no_grad():
        for _ in range(288):
            obs=normalize_observation(env.begin_step())
            if collect:collected.append(obs.clone())
            policy=obs.clone()
            if MODE!='normal':
                cols=GROUPS[MODE]
                if cols:
                    policy[:,cols]=torch.tensor(CONSTANTS['inputs'],dtype=policy.dtype)[cols]
                    # Preserve all empty-slot values and presence masks.
                    for slot in range(10):
                        empty=obs[:,slot*6]<=0.5
                        slot_cols=[j for j in cols if slot*6<=j<slot*6+6]
                        for j in slot_cols:policy[empty,j]=obs[empty,j]
            action=torch.stack([ACTORS[i](policy[i]) for i in range(20)])
            env.apply_action(action)
            enabled.append(bool(env.current_tracking_enabled))
            success.append(abs(float(env.last_raw_actor_residual_kw))<=float(env.tol_narrow_metrics))
    m=env.get_metrics();en=np.asarray(enabled).reshape(48,6);ok=np.asarray(success).reshape(48,6);blocks=en.any(1)
    row={'id':case['id'],'mode':MODE,'date':p['date'],'command':p['source'],'ev_seed':case['ev_seed'],'ev_hash':ev_hash,
         'tracking_pct':float(m['tracking_success_rate']),'soc_pct':100-float(m['soc_miss_rate']),
         'mae_kw':float(m['raw_actor_mae_kw']),'block_pass_pct':float(100*(ok|~en).all(1)[blocks].mean()),
         'assessed_steps':int(m['tracking_steps']),'departing_evs':int(m['departing_evs'])}
    row['sum']=row['tracking_pct']+row['soc_pct']
    if not all(math.isfinite(row[k])for k in ('tracking_pct','soc_pct','mae_kw','sum')):raise ValueError(row)
    return (row,collected)if collect else row

def prepare(market,episode,commands,ev_seeds,confirmation=False,excluded_sources=()):
    import numpy as np
    from training.bid_bank import BidBank
    from environment.normalize import load_observation_normalization_for_archive
    from EnvConfig import LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR
    from market.activation_scenarios import load_proxy_shape_library,_row_source,stable_seed_for_context,LOWER_CONTROL_POOL,build_activation_scenario_set,scenarios_to_solver_payload
    from training.lower_bid_training import _lower_command_seed,interim_test_command_sources,build_fixed_upper_bid_training_episode
    train_dir,val_dir,run=roots(market);train=BidBank(train_dir);val=BidBank(val_dir)
    load_observation_normalization_for_archive(run)
    lib=load_proxy_shape_library(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    sources=[_row_source(lib.iloc[i])for i in range(len(lib))]
    interim=interim_test_command_sources(val.entries,interim_test_episodes=5,library_dir=LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    pool=[s for s,d in zip(sources,lib['declared_partition'])if d in ('validation','test')and s not in interim]
    seen=set(interim)
    for idx in range(episode+25+1):
        e=train.entries[max(idx-1,0)%len(train.entries)];_,seed=_lower_command_seed(e,idx,'train')
        rng=np.random.default_rng(stable_seed_for_context(e['service_date'],seed,'explicit_5min_random',LOWER_CONTROL_POOL))
        seen.add(pool[int(rng.permutation(len(pool))[0])])
    # Validate fast reconstruction against the canonical sampler on three episodes.
    for idx in (0,25,episode+25):
        e=train.entries[max(idx-1,0)%len(train.entries)];_,seed=_lower_command_seed(e,idx,'train')
        canonical=build_activation_scenario_set(e['service_date'],1,seed,LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,LOWER_CONTROL_POOL,True,interim)[0].source
        rng=np.random.default_rng(stable_seed_for_context(e['service_date'],seed,'explicit_5min_random',LOWER_CONTROL_POOL))
        assert canonical==pool[int(rng.permutation(len(pool))[0])]
    seen.update(excluded_sources)
    prepared=[];cases=[];calib=[]
    for split,bank,indices,n_commands in [('calibration',train,[0,2,3,6,12],1),('evaluation',val,list(range(len(val.entries))),commands)]:
        for day in indices:
            b=dict(bank.load_entry(bank.entries[day]));signals=build_activation_scenario_set(b['service_date'],n_commands,int(b['forecast_seed'])+8_500_019+(1_300_007 if confirmation else 0),
                          LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,'holdout',True,seen if split=='evaluation' else None)
            for c,signal in enumerate(signals):
                fixed=dict(b);fixed.update(activation_scenario_payload=scenarios_to_solver_payload([signal]),activation_scenarios=1)
                target,tol,arr,_=build_fixed_upper_bid_training_episode(fixed,0)
                p={'target':target,'tol':np.asarray(tol),'tracking':np.asarray(arr['tracking_enabled_series'],bool),'arr':arr,'date':b['service_date'],'source':signal.source}
                k=len(prepared);prepared.append(p)
                if split=='calibration':calib.append({'id':f'calib_{day}','mode':'normal','payload':k,'ev_seed':7_000_000+day})
                else:
                    assert signal.source not in seen
                    for ev in range(ev_seeds):cases.append({'id':f'd{day}_c{c}_e{ev}','mode':'normal','payload':k,'ev_seed':(19_000_000 if confirmation else 9_000_000)+1009*day+10007*ev})
    return prepared,cases,calib,len(seen)

def run_batch(args,folder,cases,groups):
    import multiprocessing as mp
    folder.mkdir(parents=True,exist_ok=True);rowdir=folder/'rows';rowdir.mkdir(exist_ok=True)
    tasks=[dict(c,mode=m)for m in ['normal']+list(groups)for c in cases]
    rows=[];pending=[]
    for c in tasks:
        file=rowdir/f'{c["mode"]}_{c["id"]}.json'
        if file.exists():rows.append(json.loads(file.read_text()))
        else:pending.append(c)
    status={'market':args.market,'episode':args.episode,'groups':len(groups),'cases_per_group':len(cases),'expected_rows':len(tasks),'completed':len(rows),'stage':folder.name,'workers':args.workers}
    (folder/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,
                             initargs=(args.market,args.episode,PREPARED,CONSTANTS,groups))as pool:
        futures={pool.submit(rollout,c):c for c in pending}
        for f in as_completed(futures):
            r=f.result();rows.append(r);file=rowdir/f'{r["mode"]}_{r["id"]}.json'
            temp=file.with_suffix('.tmp');temp.write_text(json.dumps(r));temp.replace(file)
            status['completed']=len(rows)
            if len(rows)%20==0:(folder/'status.json').write_text(json.dumps(status,indent=2));print(folder.name,'completed',len(rows),'/',len(tasks),flush=True)
    summary=compare(rows,groups);(folder/'summary.json').write_text(json.dumps(summary,indent=2))
    status.update(stage='complete',completed=len(rows));(folder/'status.json').write_text(json.dumps(status,indent=2))
    return summary

def confirmation(args,outdir,screen):
    global PREPARED,CONSTANTS
    # This is an exploratory candidate threshold, not a scientific equivalence claim.
    def small(v):
        return v['sum_loss_day_cluster_ci95'][1]<1.0 and all(v['delta'][k]>-1.0 for k in ('tracking_pct','soc_pct','block_pass_pct'))
    candidates=[k for k,v in screen.items()if small(v)]
    (outdir/'candidate_selection.json').write_text(json.dumps({'rule':'upper 95% day-cluster interval of sum loss below 1 point, and mean losses in tracking/SoC/block rate below 1 point','candidates':candidates},indent=2))
    if not candidates:
        (outdir/'study_status.json').write_text(json.dumps({'stage':'no_low_effect_candidates','retraining':'not justified by this screen'}));return
    folder=outdir/'confirmation';folder.mkdir(exist_ok=True);file=folder/'prepared.pkl'
    if file.exists():
        with file.open('rb')as f:PREPARED,cases=pickle.load(f)
    else:
        excluded={p['source']for p in PREPARED}
        PREPARED,cases,_,_=prepare(args.market,args.episode,16,4,True,excluded)
        with file.open('wb')as f:pickle.dump((PREPARED,cases),f)
    groups={k:GROUPS[k]for k in candidates}
    # The union tests whether individually small effects conceal redundancy.
    groups['candidate_bundle']=sorted({j for k in candidates for j in GROUPS[k]})
    CONSTANTS['count_modes']=['active_EV_count']+(['candidate_bundle']if 'active_EV_count'in candidates else [])
    confirmed=run_batch(args,folder,cases,groups)
    kept=[k for k,v in confirmed.items()if small(v)]
    masks={k:{'indices':groups[k],'values':CONSTANTS['inputs'],'fixed_count':CONSTANTS['active_count']if k in CONSTANTS['count_modes']else None}for k in kept}
    (outdir/'confirmed_masks.json').write_text(json.dumps(masks,indent=2))
    (outdir/'study_status.json').write_text(json.dumps({'stage':'confirmation_complete','confirmed_candidates':kept,'next':'compare matched-seed retraining against the full-feature baseline; fixed-input tests alone do not establish dispensability'},indent=2))

def initialize(market,episode,prepared,constants,groups):
    global PREPARED,CONSTANTS,GROUPS,ACTORS
    configure(market)
    import torch
    from environment.normalize import load_observation_normalization_for_archive
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    PREPARED=prepared;CONSTANTS=constants;GROUPS=groups;ACTORS=load_actors(market,episode)
    load_observation_normalization_for_archive(roots(market)[2])

def compare(rows,groups):
    import numpy as np
    base={r['id']:r for r in rows if r['mode']=='normal'};out={}
    for mode in groups:
        rr=[r for r in rows if r['mode']==mode]
        if not rr:continue
        diffs={k:[] for k in ('tracking_pct','soc_pct','sum','mae_kw','block_pass_pct')};by_day={}
        for r in rr:
            b=base[r['id']];assert r['ev_hash']==b['ev_hash']and r['command']==b['command']and r['departing_evs']==b['departing_evs']
            for k in diffs:diffs[k].append(r[k]-b[k])
            by_day.setdefault(r['date'],[]).append(b['sum']-r['sum'])
        loss=np.array([np.mean(v)for v in by_day.values()]);rng=np.random.default_rng(20261008)
        boot=loss[rng.integers(0,len(loss),(10000,len(loss)))].mean(1)
        out[mode]={'cases':len(rr),'delta':{k:float(np.mean(v))for k,v in diffs.items()},'sum_loss_day_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
                   'per_day_sum_loss':{k:float(np.mean(v))for k,v in by_day.items()}}
    return out

def main():
    global PREPARED,CONSTANTS,GROUPS,ACTORS
    args=P.parse_args();configure(args.market)
    print('prepare_start', args.market, 'episode', args.episode, flush=True)
    import numpy as np
    import torch
    from environment.normalize import load_observation_normalization_for_archive
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES
    outdir=Path(__file__).parent/args.market;outdir.mkdir(exist_ok=True)
    prep=outdir/'prepared.pkl'
    if prep.exists():
        with prep.open('rb')as f:PREPARED,cases,calib,seen=pickle.load(f)
    else:
        PREPARED,cases,calib,seen=prepare(args.market,args.episode,args.commands,args.ev_seeds)
        with prep.open('wb')as f:pickle.dump((PREPARED,cases,calib,seen),f)
    load_observation_normalization_for_archive(roots(args.market)[2])
    torch.set_num_threads(1);GROUPS=group_indices();CONSTANTS={};ACTORS=load_actors(args.market,args.episode)
    calibration=outdir/'constants.json'
    if calibration.exists():CONSTANTS=json.loads(calibration.read_text())
    else:
        xx=[]
        for case in calib:
            _,x=rollout(case,True);xx.extend(x)
        x=torch.stack(xx);ev=x[:,:,:60].reshape(-1,10,6);mask=ev[...,0]>.5
        mu=x.reshape(-1,x.shape[-1]).mean(0)
        for j in range(1,6):mu[list(range(j,60,6))]=ev[...,j][mask].mean()
        for n in ('service_day_holiday','market_tracking_enabled'):
            j=60+LOCAL_TAIL_FEATURE_NAMES.index(n);mu[j]=float(mu[j]>=.5)
        CONSTANTS={'inputs':mu.tolist(),'active_count':float(mask.float().sum(1).mean()/10),'source':'five training-bank dates only'}
        calibration.write_text(json.dumps(CONSTANTS,indent=2))
    if args.smoke:
        r=rollout(cases[0]);r2=rollout(cases[0]);assert r==r2
        print('smoke deterministic',json.dumps(r),flush=True)
        for mode in ('EV_soc','bid_offsets_1_23','service_day_holiday','active_EV_count'):
            masked=rollout(dict(cases[0],mode=mode))
            assert masked['ev_hash']==r['ev_hash'] and masked['departing_evs']==r['departing_evs']
            print('smoke mask',mode,'sum',masked['sum'],flush=True)
        return
    mode_names=['normal']+list(GROUPS)
    tasks=[dict(c,mode=m)for m in mode_names for c in cases]
    rowdir=outdir/'rows';rowdir.mkdir(exist_ok=True)
    pending=[];rows=[]
    for c in tasks:
        file=rowdir/f'{c["mode"]}_{c["id"]}.json'
        if file.exists():rows.append(json.loads(file.read_text()))
        else:pending.append(c)
    status={'market':args.market,'episode':args.episode,'groups':len(GROUPS),'cases_per_group':len(cases),'expected_rows':len(tasks),'completed':len(rows),'stage':'screen','workers':args.workers,'excluded_training_sources':seen}
    (outdir/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,
                             initargs=(args.market,args.episode,PREPARED,CONSTANTS,GROUPS))as pool:
        futures={pool.submit(rollout,c):c for c in pending}
        for f in as_completed(futures):
            r=f.result();rows.append(r);file=rowdir/f'{r["mode"]}_{r["id"]}.json'
            temp=file.with_suffix('.tmp');temp.write_text(json.dumps(r));temp.replace(file)
            status['completed']=len(rows)
            if len(rows)%20==0:(outdir/'status.json').write_text(json.dumps(status,indent=2));print('completed',len(rows),'/',len(tasks),flush=True)
    summary=compare(rows,GROUPS);(outdir/'screen_summary.json').write_text(json.dumps(summary,indent=2))
    status.update(stage='screen_complete',completed=len(rows));(outdir/'status.json').write_text(json.dumps(status,indent=2))
    print('screen_complete',flush=True)
    confirmation(args,outdir,summary)

if __name__=='__main__':main()
