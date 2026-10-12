"""既存30日入札に対する曜日診断の指令・EV乱数を用意する。入札は解かない。"""
import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('market',choices=['aemo','ercot']);ap.add_argument('--output',required=True);a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('screen',Path(__file__).with_name('trace_screen_helpers.py'))
    screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen);screen.ROOT=ROOT;screen.configure(a.market)
    import numpy as np
    from training.bid_bank import BidBank
    from training.lower_bid_training import interim_test_command_sources,_lower_command_seed,build_fixed_upper_bid_training_episode
    from market.activation_scenarios import load_proxy_shape_library,_row_source,stable_seed_for_context,LOWER_CONTROL_POOL,build_activation_scenario_set,scenarios_to_solver_payload
    from EnvConfig import LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR
    from environment.calendars import is_public_holiday
    import datetime
    tr,va,run=screen.roots(a.market);train=BidBank(tr);valid=BidBank(va)
    lib=load_proxy_shape_library(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    sources=[_row_source(lib.iloc[i]) for i in range(len(lib))]
    interim=interim_test_command_sources(valid.entries,interim_test_episodes=5,library_dir=LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR)
    pool=[s for s,d in zip(sources,lib['declared_partition']) if d in ('validation','test') and s not in interim]
    seen=set(interim)
    # Exclude every actual training draw through the saved stop point.
    learned={'aemo':1634,'ercot':1595}[a.market]
    for idx in range(learned+25+1):
        e=train.entries[max(idx-1,0)%len(train.entries)];_,seed=_lower_command_seed(e,idx,'train')
        rng=np.random.default_rng(stable_seed_for_context(e['service_date'],seed,'explicit_5min_random',LOWER_CONTROL_POOL))
        seen.add(pool[int(rng.permutation(len(pool))[0])])
    for idx in (0,25,learned+25):
        e=train.entries[max(idx-1,0)%len(train.entries)];_,seed=_lower_command_seed(e,idx,'train')
        canonical=build_activation_scenario_set(e['service_date'],1,seed,LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,LOWER_CONTROL_POOL,True,interim)[0].source
        rng=np.random.default_rng(stable_seed_for_context(e['service_date'],seed,'explicit_5min_random',LOWER_CONTROL_POOL))
        assert canonical==pool[int(rng.permutation(len(pool))[0])]
    prepared=[];cases=[];days=[]
    for split,bank in [('train_bid',train),('validation_bid',valid)]:
        for i,entry in enumerate(bank.entries):
            b=dict(bank.load_entry(entry));day=b['service_date'];date=datetime.date.fromisoformat(day)
            signals=build_activation_scenario_set(day,2,int(b['forecast_seed'])+31_670_009,LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,'holdout',True,seen)
            days.append({'date':day,'weekday':date.weekday(),'public_holiday':is_public_holiday(date,'JP'),'bank_split':split})
            for c,signal in enumerate(signals):
                assert signal.source not in seen
                fixed=dict(b);fixed.update(activation_scenario_payload=scenarios_to_solver_payload([signal]),activation_scenarios=1)
                target,tol,arr,_=build_fixed_upper_bid_training_episode(fixed,0);k=len(prepared)
                prepared.append({'target':target,'tol':np.asarray(tol),'tracking':np.asarray(arr['tracking_enabled_series'],bool),'arr':arr,'date':day,'source':signal.source})
                for ev in range(2):cases.append({'id':f'{split}_d{i}_c{c}_e{ev}','mode':'normal','payload':k,'ev_seed':39_000_000+(1_000_000 if split=='validation_bid' else 0)+1009*i+10007*ev})
    assert len(days)==30 and len(cases)==120
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    with out.open('wb') as f:pickle.dump((prepared,cases),f,protocol=4)
    out.with_suffix('.json').write_text(json.dumps({'market':a.market,'days':days,'cases':len(cases),'commands_per_day':2,'EV_seeds_per_command':2,'excluded_training_sources':len(seen),'saved_training_episode':learned,'interpretation':'weekday associations include differences in EV arrivals and bids; these are not causal weekday effects'},indent=2),encoding='utf-8')
    print(json.dumps({'market':a.market,'dates':days,'cases':len(cases)}),flush=True)
if __name__=='__main__':main()
