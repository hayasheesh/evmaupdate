"""適用学習器の実行時情報、物理行動、更新、完全再開の検証。"""
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
import pytest
import stage_worker

stage_worker.configure('aemo','1')
from adapted_agent import AFMControlAgent
from environment.observation_config import EV_FEAT_DIM, EV_FEATURE_NAMES, LOCAL_TAIL_DIM, LOCAL_TAIL_FEATURE_NAMES
from environment.normalize import configure_observation_normalization
torch.set_num_threads(1)


def agent_and_state():
    configure_observation_normalization({'command_scale':'bank','demand_scale_kw':100.,'demand_center_kw':0.})
    torch.manual_seed(21)
    slots=10;s_dim=slots*EV_FEAT_DIM+LOCAL_TAIL_DIM
    a=AFMControlAgent(s_dim,slots,6);a.batch=4;a.buf.buf_size=16;a.warmup_steps=4
    state=torch.zeros(6,s_dim)
    block=state[:,:slots*EV_FEAT_DIM].reshape(6,slots,EV_FEAT_DIM)
    block[:,:2,0]=1
    block[:,:2,EV_FEATURE_NAMES.index('soc')]=.5
    block[:,:2,EV_FEATURE_NAMES.index('needed_soc')]=.2
    block[:,:2,EV_FEATURE_NAMES.index('remaining_time')]=.1
    block[:,:2,EV_FEATURE_NAMES.index('battery_capacity_kwh')]=.6
    state[:,slots*EV_FEAT_DIM+LOCAL_TAIL_FEATURE_NAMES.index('market_tracking_enabled')]=1
    return a,state


def fill(a,state):
    a.episode_start()
    for _ in range(6):
        action=a.act(state,noise=False)
        power=torch.zeros(6,10);power[:,:2]=1
        a.cache_experience(state,state,action,torch.ones(6),torch.tensor(.2),torch.zeros(6),
                           actual_station_powers=power.sum(1),actual_ev_power_kw=power)


def test_slot_permutation_and_padding():
    a,s=agent_and_state()
    result=a.act(s,noise=False)
    assert torch.count_nonzero(result[:,2:])==0
    order=torch.tensor([1,0,2,3,4,5,6,7,8,9])
    perm=s.clone();perm[:,:10*EV_FEAT_DIM]=s[:,:10*EV_FEAT_DIM].reshape(6,10,EV_FEAT_DIM)[:,order].reshape(6,-1)
    torch.testing.assert_close(a.act(perm,noise=False),result[:,order])


def test_global_reference_uses_fleet_state_and_respects_feasible_power():
    a,s=agent_and_state()
    first=a._contexts(s.unsqueeze(0))
    changed=s.clone();changed[1,:10*EV_FEAT_DIM].reshape(10,EV_FEAT_DIM)[:2,EV_FEATURE_NAMES.index('soc')]=1
    second=a._contexts(changed.unsqueeze(0))
    assert not torch.equal(first[:,0,:2,-1],second[:,0,:2,-1])
    assert (second[0,1,:2,-1]<=0).all()
    assert (second.abs()[...,-1]<=1+1e-6).all()


def test_ddpg_update_changes_actor_and_attention_critic():
    a,s=agent_and_state();fill(a,s)
    actor=[p.detach().clone() for p in a.actors[0].parameters()]
    critic=[p.detach().clone() for p in a.critics[0].parameters()]
    a.update()
    assert any(not torch.equal(x,y) for x,y in zip(actor,a.actors[0].parameters()))
    assert any(not torch.equal(x,y) for x,y in zip(critic,a.critics[0].parameters()))
    assert torch.isfinite(torch.tensor(a.last_global_critic_loss))
    assert a.gamma==.99 and a.tau==.005 and a.grad_clip==5


def test_resume_preserves_next_update_and_rejects_different_objective():
    a,s=agent_and_state();fill(a,s);a.update()
    saved=copy.deepcopy(a.training_resume_state_dict())
    b,_=agent_and_state();b.load_training_resume_state_dict(saved)
    torch.testing.assert_close(a.act(s,noise=False),b.act(s,noise=False),rtol=0,atol=0)
    torch.manual_seed(31);a.update()
    torch.manual_seed(31);b.update()
    for x,y in zip(a.actors[0].parameters(),b.actors[0].parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(a.critics[0].parameters(),b.critics[0].parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    b.local_reward_weight*=2
    with pytest.raises(ValueError,match='compatibility mismatch'):
        b.load_training_resume_state_dict(saved)
