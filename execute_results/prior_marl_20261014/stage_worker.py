"""共通入札を使うMARL-AFM適用実験。設定変更はこのプロセス内で行う。"""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'execute_results/resbid_6station_20261012/stage_worker.py'
spec=importlib.util.spec_from_file_location('main_settings',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
TASKS={t:(signal,f'wr_cap77wd_{t}plan_6station_afm_port') for t,(signal,_) in base.TASKS.items()}


def banks(task):return base.banks(task)


def configure(task,local_weight=None):
    weight=str(local_weight if local_weight is not None else os.environ.get('AFM_LOCAL_WEIGHT','1'))
    config=base.configure(task)
    os.environ['EVMA_AFM_LOCAL_WEIGHT']=weight
    config.MARL_ALGORITHM='marl_afm_port';config.MEMORY_SIZE=100000
    config.BATCH_SIZE=256;config.GAMMA=.99;config.TAU=.005
    config.LR_ACTOR=1e-4;config.LR_CRITIC_LOCAL=.001
    config.LR_GLOBAL_CRITIC=.001;config.GAMMA_GLOBAL=.99;config.TAU_GLOBAL=.005
    config.ACTOR_HIDDEN_SIZE=64;config.GLOBAL_CRITIC_HIDDEN_SIZE=128
    config.STD_MADDPG_CRITIC_HIDDEN=128
    config.ENV_SEED=141026
    os.environ['EVMA_ENV_SEED']='141026'
    # 元の本線の報酬形状・kW尺度を使い、適用実験のteam rewardの重みを振る。
    sys.path.insert(0,str(HERE))
    from adapted_agent import AFMControlAgent
    def factory(s_dim,max_evs_per_station,n_agent,**kwargs):
        return AFMControlAgent(s_dim,max_evs_per_station,n_agent,**kwargs)
    import training.Agent.standard_maddpg as standard
    standard.build_marl_agent=factory
    import training.system_controller as system
    system.build_marl_agent=factory
    import training.training_resume as resume
    extra=tuple(Path(__file__).parent.relative_to(ROOT)/name for name in ('stage_worker.py','adapted_agent.py','afm_components.py','method_config.json'))
    resume._PRETRAIN_SOURCE_FILES=resume._PRETRAIN_SOURCE_FILES+extra
    import tools.Utils as utils
    original_snapshot=utils.snapshot_code_to_archive
    def snapshot(model_dir,project_root=None):
        result=original_snapshot(model_dir,project_root)
        out=Path(model_dir)/'method_source';out.mkdir(exist_ok=True)
        for name in ('stage_worker.py','adapted_agent.py','afm_components.py','method_config.json'):
            if (HERE/name).exists():shutil.copy2(HERE/name,out/name)
        return result
    utils.snapshot_code_to_archive=snapshot
    if 'training.train' in sys.modules:
        sys.modules['training.train'].build_marl_agent=factory
        sys.modules['training.train'].snapshot_code_to_archive=snapshot
    return config


def main():
    task,stage=sys.argv[1:3]
    weight=sys.argv[3] if len(sys.argv)>3 else '1'
    config=configure(task,weight)
    import torch
    torch.set_num_threads(1)
    if stage=='preflight':
        from environment.EVEnv import EVEnv
        from environment.normalize import normalize_observation
        from training.system_controller import build_agent
        import numpy as np
        env=EVEnv();env.reset(net_demand_series=np.zeros(config.EPISODE_STEPS),service_date='2024-08-05')
        agent=build_agent(env);agent.set_test_mode(True)
        action=agent.act(normalize_observation(env.begin_step()),env,noise=False)
        assert action.shape==(6,env.max_ev_per_station) and torch.isfinite(action).all()
        print(json.dumps({'algorithm':config.MARL_ALGORITHM,'local_weight':weight,'action_shape':list(action.shape),'cuda':torch.cuda.is_available()}))
    elif stage=='pretrain':
        episodes=sys.argv[4] if len(sys.argv)>4 else '2000'
        model=sys.argv[5] if len(sys.argv)>5 else TASKS[task][1]+'_final'
        import pre_train
        train,test=banks(task)
        sys.argv=['pre_train.py','--episodes',episodes,'--model-name',model,'--bank-dir',str(train),
                  '--test-bank-dir',str(test),'--resume-checkpoint-interval','100',*sys.argv[6:]]
        pre_train.main()
    else:raise ValueError(stage)


if __name__=='__main__':main()
