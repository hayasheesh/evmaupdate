"""保存した20局行動器の観測入力に対する局所感度を測る。

検証用5日を同じ固定seedで動かして観測を集める。入力の標準偏差と、
在車EVの各出力の絶対微分の平均を掛ける。EV出力間の符号は相殺しない。
EV特徴は10枠分、入札の先読みは各24入力分をまとめ、合計と1入力当たりを記録する。
休日フラグは0/1を反転した出力差も記録する。性能への因果効果を表す指標ではない。
CPUだけで解析し、学習器のパラメータは更新しない。
"""
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
task, episode = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, str(HERE.parent / f'allin_20station_{task}_20261007'))
import stage_worker  # noqa: E402

days = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
config = stage_worker.configure()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import load_observation_normalization_for_archive, normalize_observation, use_instruction_scale  # noqa: E402
from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES, EV_FEATURE_NAMES  # noqa: E402
from tools.evaluator import set_env_seed  # noqa: E402
from training.bid_bank import BidBank  # noqa: E402
from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode  # noqa: E402
from training.system_controller import build_agent, find_model_path_and_episode  # noqa: E402

ROOT = HERE.parents[1]
RUN_NAMES = {'aemo': 'allin_aemoplan_AB_20station_20261007_042318', 'ercot': 'allin_ercotplan_AB_20station_20261007_042436'}
model_dir = ROOT / 'archive' / RUN_NAMES[task]
torch.set_num_threads(1)
assert not torch.cuda.is_available()
test_bank = stage_worker.TEST_BANK
bank = BidBank(test_bank)
load_observation_normalization_for_archive(model_dir)
agent = None
obs_all = []
day_records=[]
for day in days:
    fixed_bid = dict(bank.load_entry(list(bank.entries)[day]))
    payloads, mode = _activation_scenarios_for_day(fixed_bid.get('service_date'), int(fixed_bid.get('forecast_seed', 0)) + 512_209,
                                                   n_scenarios=1, scenario_partition='holdout')
    fixed_bid.update(activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
    day_records.append({"index":day,"date":fixed_bid.get("service_date"),"command_source":payloads[0].get("source"),"ev_seed":3_000_000+day})
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, 0)
    tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool)[:EPISODE_STEPS]
    set_env_seed(3_000_000 + day)
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = False
    env.use_residual_bess = False
    env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=tracking,
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=arr.get('service_date'), baseline_series=arr.get('baseline_series'))
    if agent is None:
        agent = build_agent(env)
        ckpt, ep = find_model_path_and_episode(str(model_dir), episode)
        agent.load_actors(ckpt, ep, map_location='cpu')
        agent.set_test_mode(True)
    agent.update_active_evs(env)
    for t in range(EPISODE_STEPS):
        obs = normalize_observation(env.begin_step())
        obs_all.append(obs.detach().cpu())
        agent.update_active_evs(env)
        with torch.no_grad():
            actions = agent.act(obs, env=env, noise=False)
        env.apply_action(actions)

X = torch.stack(obs_all)                       # step × station × dim
n_st, dim = X.shape[1], X.shape[2]
ev_dim = dim - len(LOCAL_TAIL_FEATURE_NAMES)
names = [f'EV:{n}' for n in EV_FEATURE_NAMES] + list(LOCAL_TAIL_FEATURE_NAMES)
# Absolute derivatives per output avoid cancellation between EV actions.
rows={};details=[];holiday_flip=[]
for st in range(n_st):
    actor=agent.actors[st].to('cpu').eval()
    data=X[:,st,:]
    presence=(data[:,:ev_dim].reshape(len(data),-1,len(EV_FEATURE_NAMES))[...,0]>0.5)
    active=presence.any(1)
    sd=data[active].std(0,unbiased=False)
    sumgrad=torch.zeros(dim);n_outputs=0
    for start in range(0,len(data),256):
        x=data[start:start+256].clone().requires_grad_(True)
        a=actor(x);mask=presence[start:start+256]
        for slot in range(a.shape[1]):
            if not bool(mask[:,slot].any()):continue
            g=torch.autograd.grad((a[:,slot]*mask[:,slot]).sum(),x,retain_graph=True)[0]
            sumgrad+=g.abs().sum(0).detach()
            n_outputs+=int(mask[:,slot].sum())
    eff=(sumgrad/max(n_outputs,1)*sd).numpy()
    groups={}
    for j,n in enumerate(EV_FEATURE_NAMES):groups['EV:'+n]=list(range(j,ev_dim,len(EV_FEATURE_NAMES)))
    for j,n in enumerate(LOCAL_TAIL_FEATURE_NAMES):
        if n.startswith('bid_') and 'lookahead' in n:
            field=n.split('_')[1];key='bid_lookahead_'+field
        else:key=n
        groups.setdefault(key,[]).append(ev_dim+j)
    for name,indices in groups.items():
        rows.setdefault(name,{'inputs':len(indices),'station_sum':[]})['station_sum'].append(float(eff[indices].sum()))
    # A binary holiday flag can also be compared by flipping just that flag.
    j=ev_dim+LOCAL_TAIL_FEATURE_NAMES.index('service_day_holiday')
    with torch.no_grad():
        changed=data.clone();changed[:,j]=1-changed[:,j]
        delta=(actor(changed)-actor(data)).abs()
        holiday_flip.append(float(delta[presence].mean()))
    details.append({'station':st,'present_outputs':n_outputs,'active_steps':int(active.sum()),'input_scores':eff.tolist()})
    print(f'{task} station {st+1}/{n_st}',flush=True)
for name,v in rows.items():
    v['mean_sum']=float(np.mean(v['station_sum']))
    v['mean_per_input']=v['mean_sum']/v['inputs']
ranked=sorted(rows.items(),key=lambda x:-x[1]['mean_per_input'])
result={'task':task,'episode':episode,'run_dir':str(model_dir),'device':'cpu','days':day_records,'steps':len(X),'stations':n_st,
        'method':'mean absolute derivative for each present EV output, multiplied by observed input population standard deviation; no cancellation between outputs',
        'groups':dict(ranked),'holiday_flip_mean_abs_action':float(np.mean(holiday_flip)),'holiday_flip_by_station':holiday_flip,'station_details':details}
out=HERE/f'feature_sensitivity20_{task}_ep{episode}.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('saved',out,flush=True)
for n,v in ranked:print(n,v['inputs'],round(v['mean_per_input'],6),round(v['mean_sum'],6),flush=True)
print('holiday_flip',float(np.mean(holiday_flip)),flush=True)
