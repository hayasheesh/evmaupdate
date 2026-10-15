"""正規化を固定し、行動器・強制充電・連系点BESSの組合せを40組で測る。"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'execute_results/paper_eval_20261014'))


def episode(job):
    task,force,kw,day,c,e,model,ep=job
    import eval_episode as t3
    t3._setup(task)
    import numpy as np
    import torch
    from EnvConfig import BESS_DISCHARGE_EFFICIENCY,BESS_MIN_SOC_PCT,BESS_TARGET_SOC_PCT
    from environment.normalize import load_observation_normalization_for_archive,normalize_observation
    from training.system_controller import build_agent,find_model_path_and_episode,apply_force_charging
    from market.bid_participation import participation_by_block
    load_observation_normalization_for_archive(model)
    fixed,command,target,tol,tracking,arr=t3._episode_inputs(task,day,c,4)
    env=t3._make_env(task,day,c,e,target,tol,tracking,arr,central=False)
    env.use_residual_bess=kw>0
    if kw>0:
        env.bess_power_limit_kw=kw
        env.bess_energy_capacity_kwh=kw*.5/((BESS_TARGET_SOC_PCT-BESS_MIN_SOC_PCT)/100*BESS_DISCHARGE_EFFICIENCY)
        env.bess_energy_kwh=env.bess_energy_capacity_kwh*BESS_TARGET_SOC_PCT/100
    agent=build_agent(env)
    checkpoint,_=find_model_path_and_episode(str(Path(model)/'results'/f'TEST{ep}'),ep)
    agent.load_actors(checkpoint,ep,map_location='cpu');agent.set_test_mode(True)
    pre=[];post=[];power=[];inference=[];full_step=[];energy_trace=[float(env.bess_energy_kwh)];t0=time.perf_counter()
    for _ in range(288):
        step_start=time.perf_counter()
        raw=env.begin_step();agent.update_active_evs(env)
        start=time.perf_counter()
        with torch.no_grad():actions=torch.as_tensor(agent.act(normalize_observation(raw),env=env,noise=False),dtype=torch.float32)
        inference.append((time.perf_counter()-start)*1000)
        if force:actions,_,_=apply_force_charging(actions,env,slack_kwh=.1)
        _,_,_,_,info=env.apply_action(actions)
        full_step.append((time.perf_counter()-step_start)*1000)
        energy_trace.append(float(env.bess_energy_kwh))
        pre.append(float(info['pre_bess_residual_kw']));post.append(float(info['post_bess_residual_kw']));power.append(float(info['bess_power_kw']))
    part=participation_by_block(fixed['up_plan'],fixed['down_plan'])
    def judge(values):
        err=np.abs(values);passed=err<=tol+1e-6
        failures=sum(not bool(np.all(passed[k*6:(k+1)*6])) for k in range(48) if part[k])
        return {'blocks_failed':int(failures),'blocks_assessed':int(np.sum(part)),
                'step_pass_pct':float(100*np.mean(passed[tracking])),'mae_kw':float(np.mean(err[tracking]))}
    m=env.metrics;dep=int(m['departing_evs'])
    row={'task':task,'force':force,'bess_kw':kw,'bess_kwh':float(env.bess_energy_capacity_kwh) if kw else 0,
         'day':day,'command':c,'ev_seed':e,'model':Path(model).name,'episode':ep,
         'soc_hit_pct':100*int(m.get('central_departing_evs_soc_met',m['departing_evs_soc_met']))/max(dep,1),
         'bess_abs_energy_kwh':float(np.sum(np.abs(power))*5/60),'max_bess_power_kw':float(np.max(np.abs(power))),
         'seconds':time.perf_counter()-t0,'inference_ms_median':float(np.median(inference)),
         'inference_ms_p95':float(np.quantile(inference,.95))}
    row.update({'full_simulation_step_ms_p95':float(np.quantile(full_step,.95)), 'full_simulation_step_ms_max':float(np.max(full_step)), 'full_step_times_ms':full_step, 'bess_energy_trace_kwh':energy_trace, 'bess_signed_power_kw':power, 'timing_scope':'begin_step + normalization + actor + optional_force + apply_action_including_bess_and_simulation_bookkeeping'})
    row.update({f'pre_{k}':v for k,v in judge(pre).items()});row.update({f'post_{k}':v for k,v in judge(post).items()})
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--episode',type=int,default=2000);ap.add_argument('--force',action='store_true')
    ap.add_argument('--sizes',default='1000');ap.add_argument('--workers',type=int,default=1)
    args=ap.parse_args()
    os.environ.update(TABLE3_SW_DIR=str(ROOT/'execute_results'/'resbid_6station_20261012'),CUDA_VISIBLE_DEVICES='-1',
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    out=HERE/f'bess6_{args.task}_{"force" if args.force else "raw"}_v2.jsonl'
    text=out.read_text(encoding='utf-8') if out.exists() else ''
    if text and not text.endswith('\n'):
        out.with_suffix('.incomplete_backup').write_text(text,encoding='utf-8')
        text=text.rsplit('\n',1)[0]+'\n' if '\n' in text else ''
        out.write_text(text,encoding='utf-8')
    rows=[json.loads(x) for x in text.splitlines()]
    if any(r['model']!=Path(args.model).name or r['episode']!=args.episode for r in rows):raise RuntimeError('BESS input model changed')
    done={(r['bess_kw'],r['day'],r['command'],r['ev_seed']) for r in rows}
    sizes=list(map(float,args.sizes.split(',')))
    jobs=[(args.task,args.force,kw,d,c,e,args.model,args.episode) for kw in sizes for d in range(5) for c in range(4) for e in range(2)
          if (kw,d,c,e) not in done]
    with mp.get_context('spawn').Pool(args.workers) as pool,out.open('a',encoding='utf-8') as f:
        for r in pool.imap_unordered(episode,jobs):f.write(json.dumps(r)+'\n');f.flush()
    (HERE/f'bess6_{args.task}_{"force" if args.force else "raw"}_v2.DONE').write_text('complete\n',encoding='utf-8')
    print(json.dumps({'task':args.task,'force':args.force,'sizes':sizes,'complete':True}),flush=True)


if __name__=='__main__':main()
