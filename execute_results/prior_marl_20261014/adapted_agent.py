"""MARL-AFMの制御アルゴリズムを、共通入札のEVEnvへ適用する学習器。

動的グループ、局所EV Actor、グループ参照電力、attention CriticとDDPGを使う。
報酬・分類閾値・参照電力の配分・ネットワーク幅は今回の適用条件である。
"""
from __future__ import annotations
import copy
import os
import torch
from torch import nn
from torch.nn import functional as F

from Config import EPISODE_STEPS, LOCAL_REWARD_SCALE, GLOBAL_REWARD_SCALE
from environment.observation_config import EV_FEAT_DIM, EV_FEATURE_NAMES, LOCAL_TAIL_DIM, LOCAL_TAIL_FEATURE_NAMES
from environment.normalize import get_observation_normalization_profile
from training.Agent.standard_maddpg import StandardMADDPG
from training.Agent.maddpg import device
from afm_components import AttentionCritic, LocalActor, ddpg_target


class EVGroupActor(nn.Module):
    def __init__(self, s_dim, slots, hidden):
        super().__init__()
        self.slots=slots
        self.policy=LocalActor(observation_dim=EV_FEAT_DIM+LOCAL_TAIL_DIM+2*EV_FEAT_DIM+2,
                               hidden_dim=hidden,action_dim=1)

    def forward(self, state, context=None):
        squeeze=state.ndim==1
        if squeeze:state=state.unsqueeze(0)
        ev=state[:,:self.slots*EV_FEAT_DIM].reshape(-1,self.slots,EV_FEAT_DIM)
        active=ev[...,0]>.5
        tail=state[:,self.slots*EV_FEAT_DIM:].unsqueeze(1).expand(-1,self.slots,-1)
        if context is None:
            context=ev.new_zeros(ev.shape[0],self.slots,2*EV_FEAT_DIM+2)
        result=self.policy(torch.cat([ev,tail,context],-1),active).squeeze(-1)
        return result.squeeze(0) if squeeze else result


class AFMControlAgent(StandardMADDPG):
    def __init__(self,s_dim,max_evs_per_station,n_agent,**kwargs):
        kwargs.update(lr_a=1e-4,lr_c=1e-3,tau=.005,batch=256)
        super().__init__(s_dim,max_evs_per_station,n_agent,**kwargs)
        self.gamma=.99
        self.local_reward_weight=float(os.environ.get('EVMA_AFM_LOCAL_WEIGHT','1'))*LOCAL_REWARD_SCALE
        self.global_reward_weight=GLOBAL_REWARD_SCALE
        self.group_count=self.n*3
        self.hidden=128;self.heads=4
        self.critic_hidden=self.hidden
        self.urgent_steps=12
        self.actors=[EVGroupActor(s_dim,self.max_evs,64).to(device) for _ in range(self.n)]
        self.t_actors=[copy.deepcopy(a) for a in self.actors]
        self.opt_a=[torch.optim.Adam(a.parameters(),lr=1e-4) for a in self.actors]
        critic=AttentionCritic(aggregate_dim=LOCAL_TAIL_DIM+EV_FEAT_DIM+2,
                               group_dim=2*EV_FEAT_DIM+2,hidden_dim=self.hidden,heads=self.heads).to(device)
        self.critics=[critic];self.t_critics=[copy.deepcopy(critic)]
        self.opt_c=[torch.optim.Adam(critic.parameters(),lr=1e-3)]
        self.global_critic1=critic;self.t_global_critic1=self.t_critics[0]
        self.opt_global_c1=self.opt_c[0]
        self.buf.buf_size=100000
        self.epsilon_initial=self.epsilon_final=self.epsilon=0

    def _features(self,state):
        ev=state[:,:,:self.max_evs*EV_FEAT_DIM].reshape(-1,self.n,self.max_evs,EV_FEAT_DIM)
        active=ev[...,0]>.5
        needed=ev[...,EV_FEATURE_NAMES.index('needed_soc')]
        remaining=ev[...,EV_FEATURE_NAMES.index('remaining_time')]*EPISODE_STEPS
        category=torch.where((needed>0)&(remaining<=self.urgent_steps),0,
                             torch.where(needed>0,1,2))
        station=torch.arange(self.n,device=state.device).view(1,self.n,1)
        ids=(3*station+category).reshape(state.shape[0],-1)
        mask=active.reshape(state.shape[0],-1)
        flat=ev.reshape(state.shape[0],-1,EV_FEAT_DIM).masked_fill(~mask.unsqueeze(-1),0)
        membership=F.one_hot(ids,self.group_count).to(state.dtype)*mask.unsqueeze(-1)
        count=membership.sum(1)
        mean=torch.bmm(membership.transpose(1,2),flat)/count.clamp_min(1).unsqueeze(-1)
        maximum=torch.stack([flat.masked_fill(~(mask&(ids==k)).unsqueeze(-1),-torch.inf).amax(1)
                             for k in range(self.group_count)],1)
        maximum=maximum.masked_fill((count==0).unsqueeze(-1),0)
        return ev,active,ids,membership,count,mean,maximum

    def _contexts(self,state):
        ev,active,ids,member,count,mean,maximum=self._features(state)
        soc,capacity,power=self._extract_ev_physics(ev)
        _,kw_per_soc,_=self._ev_physics_factors(capacity,power)
        lower=torch.maximum(-power,-soc*kw_per_soc).masked_fill(~active,0)
        upper=torch.minimum(power,(100-soc)*kw_per_soc).masked_fill(~active,0)
        tail=state[:,:,self.max_evs*EV_FEAT_DIM:]
        profile=get_observation_normalization_profile()
        demand=tail[:,0,LOCAL_TAIL_FEATURE_NAMES.index('demand_0')]*float(profile['demand_scale_kw'])+float(profile['demand_center_kw'])
        total_lower=lower.sum((1,2));span=(upper-lower).sum((1,2))
        fraction=((demand-total_lower)/span.clamp_min(1e-6)).clamp(0,1).view(-1,1,1)
        # 集約側は実行可能範囲の同じ割合を各EVの参照電力にする。
        # Actor自身の行動をこの参照電力へ置き換える処理はしない。
        ref=(lower+fraction*(upper-lower))/power.clamp_min(1e-6)
        selected=torch.gather(torch.cat([mean,maximum,count.unsqueeze(-1)/self.max_evs],-1),1,
                              ids.unsqueeze(-1).expand(-1,-1,2*EV_FEAT_DIM+1))
        context=torch.cat([selected,ref.reshape(state.shape[0],-1,1)],-1)
        return context.reshape(state.shape[0],self.n,self.max_evs,-1)

    def _policy(self,state,target=False):
        context=self._contexts(state)
        actors=self.t_actors if target else self.actors
        return torch.stack([actor(state[:,i,:],context[:,i]) for i,actor in enumerate(actors)],1)

    def _q(self,state,actions,target=False):
        ev,active,ids,member,count,mean,maximum=self._features(state)
        flat_action=actions.reshape(state.shape[0],-1).masked_fill(~active.reshape(state.shape[0],-1),0)
        avg_action=torch.bmm(member.transpose(1,2),flat_action.unsqueeze(-1))/count.clamp_min(1).unsqueeze(-1)
        groups=torch.cat([mean,maximum,avg_action,count.unsqueeze(-1)/self.max_evs],-1)
        total=count.sum(1).clamp_min(1)
        aggregate_mean=(mean*count.unsqueeze(-1)).sum(1)/total.unsqueeze(-1)
        common=state[:,0,self.max_evs*EV_FEAT_DIM:]
        aggregate=torch.cat([common,aggregate_mean,(flat_action.sum(1)/total).unsqueeze(-1),
                             (count.sum(1)/(self.n*self.max_evs)).unsqueeze(-1)],-1)
        critic=self.t_critics[0] if target else self.critics[0]
        return critic(aggregate,groups,count==0)[0]

    def act(self,state,env=None,noise=True):
        state=torch.as_tensor(state,dtype=torch.float32,device=device)
        block=state[:,:self.max_evs*EV_FEAT_DIM].reshape(self.n,self.max_evs,EV_FEAT_DIM)
        active=block[...,0]>.5
        self.active_slot_mask=active;self.active_evs=active.sum(1);self.active_evs_tensor=self.active_evs
        if env is not None:self.env=env
        with torch.no_grad():
            action=self._policy(state.unsqueeze(0)).squeeze(0)
            if noise and not self.test_mode and self.ou_noise_scale>0:
                action=action+self.ou_noise_scale*self.ou_noise.sample(active)
        return action.clamp(-1,1).masked_fill(~active,0)

    def update(self):
        if self.test_mode or self.buf.size<max(self.warmup_steps,self.batch):
            self._zero_update_logs();return
        self.update_step+=1
        (s,s2,a,local,_global,d,station_power,ev_power,global_r,_,_,_)=self.buf.sample_with_nstep_global(self.batch,1,self.gamma)
        ctx=self._build_update_ctx(s,s2,a,local,d,station_power,ev_power)
        actual=ctx['a_actual'].masked_fill(ctx['ev_padding_mask'],0)
        team_reward=self.global_reward_weight*global_r.reshape(-1,1)+self.local_reward_weight*local.mean(1,keepdim=True)
        with torch.no_grad():
            next_action=self._executed_actions(self._policy(s2,target=True),ctx['current_socs_s2'],ctx['max_power_kw_s2'],
                ctx['soc_step_per_kw_s2'],ctx['kw_per_soc_step_s2'],ctx['inv_max_power_kw_s2'],ctx['ev_padding_mask_s2'],False)
            target=ddpg_target(team_reward,d.bool().all(1,keepdim=True),self._q(s2,next_action,target=True),gamma=self.gamma)
        predicted=self._q(s,actual)
        loss=F.mse_loss(predicted,target)
        if not torch.isfinite(loss):raise FloatingPointError('AFM critic loss is not finite')
        self.opt_c[0].zero_grad();loss.backward()
        critic_norm=self._clip_and_measure(self.critics[0])
        if not torch.isfinite(critic_norm):raise FloatingPointError('AFM critic gradient is not finite')
        self.opt_c[0].step()
        for p in self.critics[0].parameters():p.requires_grad_(False)
        try:
            for opt in self.opt_a:opt.zero_grad()
            policy_action=self._executed_actions(self._policy(s),ctx['current_socs'],ctx['max_power_kw'],
                ctx['soc_step_per_kw'],ctx['kw_per_soc_step'],ctx['inv_max_power_kw'],ctx['ev_padding_mask'],True)
            actor_loss=-self._q(s,policy_action).mean()
            if not torch.isfinite(actor_loss):raise FloatingPointError('AFM actor loss is not finite')
            actor_loss.backward()
            norms=torch.stack([self._clip_and_measure(actor) for actor in self.actors])
            if not torch.isfinite(norms).all():raise FloatingPointError('AFM actor gradient is not finite')
            for opt in self.opt_a:opt.step()
        finally:
            for p in self.critics[0].parameters():p.requires_grad_(True)
        self._polyak_update_targets()
        self.actor_norms_before_clip=norms.cpu().tolist()
        self.actor_norms=[min(x,self.grad_clip) for x in self.actor_norms_before_clip]
        self.actor_losses=[float(actor_loss.detach())]*self.n
        self.critic_losses=[];self.critic_norms=[0.]*self.n
        self._ep_q_raw_global.append(predicted.detach().mean())
        self._aggregate_update_logs(0,sum(x>self.grad_clip for x in self.actor_norms_before_clip))
        self.last_critic_loss=self.last_global_critic_loss=float(loss.detach())
        self.last_global_critic_grad_norm_before_clip=float(critic_norm)
        self.last_global_critic_grad_norm=min(float(critic_norm),self.grad_clip)
        self.last_global_critic_clip_count=int(float(critic_norm)>self.grad_clip)
        self.last_global_td_target_abs_mean=float(target.abs().mean())

    def _training_resume_compatibility(self):
        return dict(super()._training_resume_compatibility(),group_count=self.group_count,
                    hidden=self.hidden,heads=self.heads,urgent_steps=self.urgent_steps,
                    reference='fraction_of_feasible_fleet_range',actor='per_station_shared_EV_policy',
                    objective='global_plus_mean_station_local')
