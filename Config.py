"""
Config.py - central configuration module for the project.

This file collects hyperparameters and runtime settings for training,
the environment, exploration, and output handling in one place.
Other modules import `Config` and reference these constants directly.
"""
import os
import torch

# --- Core settings ---
# Unset, two runs of the same command differ well beyond floating point: the
# same episode came back at 71.6% and 63.4% SoC, 205 and 220 kW of raw actor
# error. Set it to an integer and `set_env_seed` seeds random, numpy and torch,
# which makes a run repeatable without making it less varied -- the generator
# still advances across episodes, it just starts from the same place.
# The actor mean-pools its EV tokens and throws the divisor away, so the same
# vehicles at the same state of charge produce the same per-vehicle action
# whether one is plugged in or ten.  A station contributes count x per-vehicle
# power, so without the count it cannot aim its total.  Setting this feeds the
# active count into the action head; the count is already computed inside the
# actor as the pooling divisor, so nothing about the observation changes.
# The mixer centres the station utilities before weighting them, so whatever is
# common to all stations leaves through the bounded bias and nothing else.  With
# seven identical stations that means the aggregate state of charge cannot move
# the global value at all.  A learned, positive share of the station mean is
# added back, leaving the centred term -- and the per-station credit it carries
# -- exactly as it was.  Over 600 episodes against the same run without it, raw
# tracking error was 29.9 kW vs 35.5 kW and no checkpoint overlapped.  Clear the
# variable for the centred-only mixer.
GLOBAL_CRITIC_KEEP_COMMON_MODE =bool (int (os .environ .get ("EVMA_GLOBAL_CRITIC_COMMON_MODE","1")))
ACTOR_USE_ACTIVE_EV_COUNT =bool (int (os .environ .get ("EVMA_ACTOR_EV_COUNT","1")))
ENV_SEED =(
lambda v :None if not str (v ).strip ()else int (v )
)(os .environ .get ("EVMA_ENV_SEED",""))  # Fixed environment random seed for reproducibility.

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")  # Prefer CUDA when available; otherwise use CPU.
PROJECT_ROOT =os .path .dirname (os .path .abspath (__file__ ))  # Absolute path to the project root directory.

# Environment, demand, arrival, and EV-scenario settings live in EnvConfig.
# Re-export them here so older modules importing from Config keep working.
from EnvConfig import *


# --- Losses and gradients ---
SMOOTHL1_BETA =0.05  # SmoothL1 beta: L2 below this threshold, L1 above it.
# The clip threshold is in units of gradient norm, which scales with the Q
# magnitude, and Q scales as 1/(1-gamma).  Raising a discount without raising
# these turns the update into a direction-only step.
GRAD_CLIP_MAX =float (os .environ .get ("EVMA_GRAD_CLIP_MAX","5.0"))  # Gradient clipping norm limit for local actor/critic.
GRAD_CLIP_MAX_GLOBAL =float (os .environ .get ("EVMA_GRAD_CLIP_MAX_GLOBAL","5.0"))  # Gradient clipping norm limit for the global critic.
BIAS_GRAD_CLIP_MAX =0.1  # Separate clipping limit for bias parameters.


# --- Training hyperparameters ---
# Optimizer and exploration settings for the current bid-bank training.
NUM_EPISODES =(int (os .environ ["EVMA_NUM_EPISODES"])if os .environ .get ("EVMA_NUM_EPISODES")else None )  # None = run until manually stopped; override with EVMA_NUM_EPISODES.
BATCH_SIZE =512
# Discounts.  At a 5-min step, 1/(1-gamma) is the effective horizon in steps:
# 0.95 -> 20 steps (1.7 h), 0.985 -> 67 steps (5.6 h), 0.99 -> 100 steps (8.3 h).
GAMMA =float (os .environ .get ("EVMA_GAMMA","0.985"))  # Local-critic discount (long SoC horizon).
GAMMA_GLOBAL =float (os .environ .get ("EVMA_GAMMA_GLOBAL","0.95"))  # Global-critic discount (near-myopic dispatch tracking).
TAU =float (os .environ .get ("EVMA_TAU","0.001"))  # Standard TD3 target-network rate.

TAU_GLOBAL =float (os .environ .get ("EVMA_TAU_GLOBAL","0.001"))
ACTOR_HIDDEN_SIZE =256

# The critic learning rate is three times the actor learning rate.
LR_ACTOR =float (os .environ .get ("EVMA_LR_ACTOR","1e-6"))
LR_CRITIC_LOCAL =float (os .environ .get ("EVMA_LR_CRITIC_LOCAL","3e-6"))
LOCAL_CRITIC_HIDDEN_SIZE =256
# Multiplier on the station reward in the local critics' TD target. The actor
# adds the local and global Q gradients unnormalized, so this sets how much the
# SoC objective weighs against tracking without touching Q_MIX_GLOBAL_WEIGHT.
# The environment's reward and its logs are unchanged.
LOCAL_REWARD_SCALE =float (os .environ .get ("EVMA_LOCAL_REWARD_SCALE","1.0"))
# 5e5 is the archive/90100 replay scale. The buffer is preallocated on the GPU,
# about 12.1 KB per transition at 20 stations; 3.5e5 lets the AEMO and ERCOT
# 20-station runs share one 16 GB card, and every run at 20 stations or more
# uses it.
MEMORY_SIZE =int (3.5e5 )if NUM_STATIONS >=20 else int (5e5 )

# One full pass over the 25-day bid bank before updates start.
WARMUP_STEPS =int (os .environ .get ("EVMA_WARMUP_STEPS","7000"))

# Balanced replay for warm-start fine-tuning. A checkpoint saves its most
# recent transitions so a fine-tune run can inherit them as an offline region
# instead of refitting the restored critic on a nearly empty online buffer.
REPLAY_SNAPSHOT_MAX_TRANSITIONS =int (
os .environ .get ("EVMA_REPLAY_SNAPSHOT_MAX_TRANSITIONS","20000")
)
# ~58 MB per snapshot at 20000 transitions x 7 stations. Only the newest is
# ever used to warm-start a fine-tune, and older ones are pruned on write, so
# this stays bounded. Set to 0 to skip saving snapshots entirely.
REPLAY_SNAPSHOT_SAVE_ENABLE =os .environ .get (
"EVMA_REPLAY_SNAPSHOT_SAVE","1"
).lower ()in ("1","true","yes","on")
# Offline share of each minibatch at the start of fine-tuning, annealed to the
# final share over BALANCED_REPLAY_DECAY_STEPS online transitions.
BALANCED_REPLAY_RATIO_INITIAL =float (
os .environ .get ("EVMA_BALANCED_REPLAY_RATIO_INITIAL","0.5")
)
BALANCED_REPLAY_RATIO_FINAL =float (
os .environ .get ("EVMA_BALANCED_REPLAY_RATIO_FINAL","0.25")
)
BALANCED_REPLAY_DECAY_STEPS =int (
os .environ .get ("EVMA_BALANCED_REPLAY_DECAY_STEPS","20000")
)

LR_GLOBAL_CRITIC =float (os .environ .get ("EVMA_LR_GLOBAL_CRITIC","3e-6"))
GLOBAL_CRITIC_HIDDEN_SIZE =256

# Independent replay batches per environment transition after warmup.
TRAIN_UPDATES_PER_ENV_STEP =max (1 ,int (os .environ .get ("EVMA_TRAIN_UPDATES_PER_STEP","1")))

# --- TD3-related settings ---
TD3_SIGMA_GLOBAL =0.20  # Standard deviation of target policy smoothing noise for the global critic.
TD3_CLIP_GLOBAL =0.7  # Clipping range for global target policy smoothing noise.
POLICY_DELAY =2  # Update the actor once every N critic updates.


# --- Global critic ---
# Station utilities are mixed by a QMIX-style global mixer.
GLOBAL_CRITIC_USE_TWIN =True  # Twin global critics with min(Q1, Q2), i.e. TD3-style global critic target.


# --- Q drift mitigation ---
# Soft bound on GlobalMLPCritic mixer bias `b` via
# MIXER_B_MAX * tanh(b_raw / MIXER_B_MAX).
# Bounding the bias keeps the global value scale controlled while leaving enough
# headroom for the Bellman fixed point of the dispatch-tracking reward.
MIXER_B_MAX =float (os .environ .get ("EVMA_MIXER_B_MAX","50.0"))
MIXER_B_MAX_ENABLE =True


# --- Gradient blending ---
# actor_grad = (1 - w) * local_Q_grad + w * global_Q_grad
Q_MIX_GLOBAL_WEIGHT =float (os .environ .get ("EVMA_Q_MIX_GLOBAL_WEIGHT","0.5"))  # Weight assigned to the global-Q gradient.


# --- Learning algorithm ---
# "hybrid" is the research learner configured above: local and global critics,
# a QMIX-style mixer, a blended actor gradient, TD3-style twin critics with
# target smoothing and a delayed actor. "maddpg" is MADDPG as in Lowe et al.
# (2017), kept for the comparison the paper has to make; see
# training/Agent/standard_maddpg.py for what it changes and what it holds fixed.
MARL_ALGORITHM =os .environ .get ("EVMA_MARL_ALGORITHM","hybrid").strip ().lower ()
# MADDPG has one discount; 0.95 is the value of Lowe et al.'s implementation.
STD_MADDPG_GAMMA =float (os .environ .get ("EVMA_STD_MADDPG_GAMMA","0.95"))
# Each station's reward: w_local * its local reward + w_global * the shared tracking reward.
STD_MADDPG_LOCAL_REWARD_WEIGHT =float (os .environ .get ("EVMA_STD_MADDPG_LOCAL_REWARD_WEIGHT","1.0"))
STD_MADDPG_GLOBAL_REWARD_WEIGHT =float (os .environ .get ("EVMA_STD_MADDPG_GLOBAL_REWARD_WEIGHT","1.0"))
STD_MADDPG_CRITIC_HIDDEN =int (os .environ .get ("EVMA_STD_MADDPG_CRITIC_HIDDEN",str (GLOBAL_CRITIC_HIDDEN_SIZE )))
STD_MADDPG_GRAD_CLIP =float (os .environ .get ("EVMA_STD_MADDPG_GRAD_CLIP",str (GRAD_CLIP_MAX )))
USE_VECTORIZED_GLOBAL_ACTOR_UPDATE =True  # Compute the shared global actor graph once and split gradients by station.


# --- Exploration noise (action-space) ---
# The MADDPG agent uses independent Gaussian exploration noise per station and EV slot.

# MADDPG uses OU_SIGMA and OU_CLIP as Gaussian scale and clipping.
OU_THETA =0.15
OU_SIGMA =float (os .environ .get ("EVMA_OU_SIGMA","0.5"))
OU_DT =1.0
OU_INIT_X =0.0
OU_CLIP =1.0

# Multiplier on the noise sample (applied to either OU or Gaussian).
# Effective per-step action std in [-1,1] = OU_NOISE_GAIN * scale * OU_SIGMA.
# Noise amplitude and episode schedule used by the current MADDPG training.
OU_NOISE_GAIN =float (os .environ .get ("EVMA_OU_NOISE_GAIN","1.0"))

# A shared noise component across all slots, exploring net charge/discharge
# directions for the global critic. Off: the AB runs at every station count use
# uncorrelated per-slot noise.
GLOBAL_CORRELATED_NOISE_GAIN =float (os .environ .get ("EVMA_GLOBAL_CORRELATED_NOISE_GAIN","0.0"))

OU_NOISE_START_EPISODE =1
OU_NOISE_END_EPISODE =int (os .environ .get ("EVMA_OU_NOISE_END_EPISODE","1500"))
OU_NOISE_SCALE_INITIAL =1.0
OU_NOISE_SCALE_FINAL =0.20


# --- Epsilon-greedy exploration ---
# Each EV slot independently rolls epsilon-greedy exploration. This is the
# long-tail anti-fixation jitter: a tiny but nonzero per-slot epsilon ensures
# even "forgotten" EVs occasionally get a random push, preventing any slot from
# being permanently neglected by the deterministic policy.
EPSILON_START_EPISODE =1
EPSILON_END_EPISODE =int (os .environ .get ("EVMA_EPSILON_END_EPISODE","1000"))
EPSILON_INITIAL =1.0
EPSILON_FINAL =0.005

# Legacy fine-tune restores a converged policy, so it must not re-run the
# pretrain's exploration schedule.  The warm-start path does not restore the
# episode counter, so epsilon runs on its own endpoints across the fine-tune's
# own budget.  The initial value is not validated: it is ten times the floor,
# enough to probe one day's specifics without discarding the policy being
# adapted.
FINETUNE_EPSILON_INITIAL =float (os .environ .get ("EVMA_FINETUNE_EPSILON_INITIAL","0.05"))
FINETUNE_EPSILON_FINAL =float (os .environ .get ("EVMA_FINETUNE_EPSILON_FINAL",str (EPSILON_FINAL )))
# 0 means "decay across the fine-tune budget", the only length that finishes
# inside the run it belongs to.
FINETUNE_EPSILON_END_EPISODE =int (os .environ .get ("EVMA_FINETUNE_EPSILON_END_EPISODE","0"))
RANDOM_ACTION_RANGE =(-1 ,1 )


# --- Output and evaluation ---
TRAIN_INTERIM_CSV_INTERVAL_EPISODES =max (1 ,int (os .environ .get (
"EVMA_TRAIN_INTERIM_INTERVAL","20")))  # Episode interval for interim test/CSV output.
# A long run does not need the same cadence as a short one: each interim
# test is five held-out episodes plus a checkpoint write.
TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES =int (
os .environ .get ("EVMA_TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES","100")
)  # Detailed TEST* graphs; fine-tune overrides this to 50.
INTERIM_TEST_EPISODES =5  # Held-out demand episodes per interim test. Keep 5 for comparable validation.
INTERIM_TEST_SEED =910_000  # Reuse identical EV realizations at every checkpoint.
INTERIM_TEST_ENABLE_PNG =True  # Write detailed rollout plots at the graph interval.
INTERIM_TEST_SAVE_DETAIL_FILES =True  # Detail collection is active only when PNG is enabled.
INTERIM_TEST_ENABLE_HISTORY_PNG =True  # Refresh compact learning-curve PNGs at the graph interval.
INTERIM_TEST_VERBOSE =True  # Print per-heldout-episode test metrics during training.

# Full diagnostics mode keeps gradient/Q diagnostics, per-episode detailed
# histories, and PNG generation so tracking/SoC failures are visible
# immediately.  Set TRAIN_FAST_PERFORMANCE_ONLY=True only for speed-only runs.
TRAIN_FAST_PERFORMANCE_ONLY =False
TRAIN_ENABLE_DIAGNOSTICS =True
TRAIN_SAVE_EPISODE_DETAIL =False
TRAIN_SAVE_SUMMARY_PNG =True
AUTO_LAUNCH_TENSORBOARD =False
CREATE_AGENT_RUNS_WRITER =True
TRAIN_FINITE_CHECK_INTERVAL_STEPS =0  # 0 = only check test/episode-end paths; avoids periodic GPU sync during training.
TRAIN_WRITE_GRAD_HEALTH =True  # Episode-end lightweight grad/loss health scalars.

TB_VERBOSE =True
