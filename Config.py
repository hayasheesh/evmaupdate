"""
Config.py - central configuration module for the project.

This file collects hyperparameters and runtime settings for training,
the environment, exploration, and output handling in one place.
Other modules import `Config` and reference these constants directly.
"""
import os
import torch

# --- Core settings ---
# Unset, two runs of the same command differ well beyond floating point. Set it
# to an integer and `set_env_seed` seeds random, numpy and torch, which makes a
# run repeatable without making it less varied -- the generator still advances
# across episodes, it just starts from the same place.
ENV_SEED =(
lambda v :None if not str (v ).strip ()else int (v )
)(os .environ .get ("EVMA_ENV_SEED",""))  # Fixed environment random seed for reproducibility.

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")  # Prefer CUDA when available; otherwise use CPU.
PROJECT_ROOT =os .path .dirname (os .path .abspath (__file__ ))  # Absolute path to the project root directory.

# Environment, demand, arrival, and EV-scenario settings live in EnvConfig.
# Re-export them here so modules importing from Config keep working.
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
# The buffer is preallocated on the GPU, about 12.1 KB per transition at 20
# stations; 3.5e5 lets two 20-station runs share one 16 GB card.
MEMORY_SIZE =int (3.5e5 )if NUM_STATIONS >=20 else int (5e5 )

# One full pass over the 25-day bid bank before updates start.
WARMUP_STEPS =int (os .environ .get ("EVMA_WARMUP_STEPS","7000"))

LR_GLOBAL_CRITIC =float (os .environ .get ("EVMA_LR_GLOBAL_CRITIC","3e-6"))
GLOBAL_CRITIC_HIDDEN_SIZE =256

# Independent replay batches per environment transition after warmup.
TRAIN_UPDATES_PER_ENV_STEP =max (1 ,int (os .environ .get ("EVMA_TRAIN_UPDATES_PER_STEP","1")))

# --- TD3-related settings ---
TD3_SIGMA_GLOBAL =0.20  # Standard deviation of target policy smoothing noise.
TD3_CLIP_GLOBAL =0.7  # Clipping range for target policy smoothing noise.
POLICY_DELAY =2  # Update the actor once every N critic updates.


# --- Global critic ---
# Station utilities are mixed by a QMIX-style global mixer, with twin critics
# and a min(Q1, Q2) target. The mixer bias is bounded as
# MIXER_B_MAX * tanh(b_raw / MIXER_B_MAX), which keeps the global value scale
# controlled while leaving headroom for the Bellman fixed point of the
# dispatch-tracking reward.
MIXER_B_MAX =float (os .environ .get ("EVMA_MIXER_B_MAX","50.0"))


# --- Gradient blending ---
# actor_grad = (1 - w) * local_Q_grad + w * global_Q_grad. w = 1 trains the
# actors from the global critic alone (the local critics are not updated).
Q_MIX_GLOBAL_WEIGHT =float (os .environ .get ("EVMA_Q_MIX_GLOBAL_WEIGHT","0.5"))  # Weight assigned to the global-Q gradient.
if not 0.0 <Q_MIX_GLOBAL_WEIGHT <=1.0 :
    raise ValueError ("EVMA_Q_MIX_GLOBAL_WEIGHT must be in (0, 1]")


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


# --- Exploration noise (action-space) ---
# Independent Gaussian noise per station and EV slot, scale OU_SIGMA, clipped at
# OU_CLIP. Effective per-step action std in [-1, 1] = OU_NOISE_GAIN * scale * OU_SIGMA,
# with the scale annealed from OU_NOISE_SCALE_INITIAL to OU_NOISE_SCALE_FINAL.
OU_SIGMA =float (os .environ .get ("EVMA_OU_SIGMA","0.5"))
OU_CLIP =1.0
OU_NOISE_GAIN =float (os .environ .get ("EVMA_OU_NOISE_GAIN","1.0"))

OU_NOISE_START_EPISODE =1
OU_NOISE_END_EPISODE =int (os .environ .get ("EVMA_OU_NOISE_END_EPISODE","1500"))
OU_NOISE_SCALE_INITIAL =1.0
OU_NOISE_SCALE_FINAL =0.20


# --- Epsilon-greedy exploration ---
# Each EV slot independently rolls epsilon-greedy exploration. A small nonzero
# per-slot epsilon keeps every slot receiving an occasional random action.
EPSILON_START_EPISODE =1
EPSILON_END_EPISODE =int (os .environ .get ("EVMA_EPSILON_END_EPISODE","1000"))
EPSILON_INITIAL =1.0
EPSILON_FINAL =0.005
RANDOM_ACTION_RANGE =(-1 ,1 )


# --- Output and evaluation ---
TRAIN_INTERIM_CSV_INTERVAL_EPISODES =max (1 ,int (os .environ .get (
"EVMA_TRAIN_INTERIM_INTERVAL","20")))  # Episode interval for interim test/CSV output.
# Each interim test is five held-out episodes plus a checkpoint write; the
# detailed TEST* graphs come at this longer interval.
TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES =int (
os .environ .get ("EVMA_TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES","100")
)
INTERIM_TEST_EPISODES =5  # Held-out demand episodes per interim test. Keep 5 for comparable validation.
INTERIM_TEST_SEED =910_000  # Reuse identical EV realizations at every checkpoint.

TB_VERBOSE =True
