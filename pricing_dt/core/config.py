"""Configuration for the offline pricing Decision Transformer experiments.

Two presets are provided:
  - smoke(): tiny, CPU-friendly, runs end-to-end in a couple of minutes. Used to
    verify the pipeline. Numbers are NOT meaningful at this scale.
  - full():  dissertation-scale defaults intended for a single GPU. Scale further
    via the CLI if compute allows.

All randomness is controlled by `seed`; experiment runners seed numpy + torch.
"""
from dataclasses import dataclass, field
from typing import List


# `slots=True` on every config class. Without it a misspelt field is a silent no-op:
# `cfg.data.n_train_episodes = 100` merely adds an attribute nothing reads. With slots
# it raises.
@dataclass(slots=True)
class SimConfig:
    """Reference-price finite-horizon pricing MDP.

    Demand is a logit in price with a reference-price effect, so price at t
    affects the demand state at t+1 -> genuine sequential structure -> stitching
    is a meaningful multi-step phenomenon.
    """
    horizon: int = 8                 # H: steps per episode
    n_prices: int = 11               # size of the discrete price grid (action space)
    p_min: float = 0.5
    p_max: float = 2.0
    n_ref_bins: int = 21             # discretisation of reference price for exact ground truth
    market_size: float = 100.0       # M
    alpha: float = 2.0               # base utility
    beta: float = 2.0                # own-price sensitivity (>0 => demand decreasing in price)
    delta: float = 3.0               # reference-price effect ((ref - price) raises utility)
    eta: float = 0.5                 # reference update speed: ref' = (1-eta)*ref + eta*price
    demand_noise: float = 0.1        # multiplicative noise std on realised demand (data only)
    seed: int = 0


@dataclass(slots=True)
class DataConfig:
    n_train_traj: int = 400          # logged trajectories for training
    n_eval_episodes: int = 256       # initial states for exact policy evaluation
    behaviour_temp: float = 0.7      # softmax temperature for stochastic behaviour policies
    epsilon: float = 0.15            # exploration of the logger
    expert_q: float = 1.0            # region-specialist quality in-region. 1.0 keeps the optimal action-pieces IN the
                                     # data so a (imitation-based) DT can stitch them; <1.0 was tried to lower the C1
                                     # per-start ceiling but it removes optimal actions the DT cannot reinvent, which
                                     # hurts every variant — see data.RegionSpecialised. Kept as a documented knob.
    # non-stationary logging (E3)
    n_segments: int = 4
    traj_per_segment: int = 150


@dataclass(slots=True)
class ModelConfig:
    d_model: int = 64
    n_layer: int = 3                 # 2 layers under-fit the return-conditioned policy (needed for stitching)
    n_head: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20                 # 8 was under-trained; the DT needs this to learn return-conditioning, not just BC
    gamma: float = 1.0               # undiscounted finite-horizon returns
    # value-based baselines (CQL / Q-DT FQI)
    q_hidden: int = 128
    q_epochs: int = 30
    cql_alpha: float = 1.0
    # structured relabelling
    relabel_lambda: float = 1.0      # 1.0 => fully structured target; <1 blends with logged RTG
    demand_epochs: int = 300         # minibatch log-space passes; 60 was under-trained (model never converged)
    elasticity_lo: float = 0.5       # bounded-elasticity range used by the structured prior
    elasticity_hi: float = 4.0


@dataclass(slots=True)
class ExpConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # E2 factorial axes
    data_sizes: List[int] = field(default_factory=lambda: [100, 400, 1600])
    noise_levels: List[float] = field(default_factory=lambda: [0.05, 0.2, 0.5])
    suboptimality: List[float] = field(default_factory=lambda: [0.05, 0.2, 0.5])  # logger epsilon
    # E3 drift sweep (softmax temperature of the snapshot logger; higher => more drift coverage)
    drift_levels: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    # C0 sequential-necessity: scan reference-price strength (delta). delta=0 => no
    # intertemporal coupling (a disguised contextual bandit); larger => more sequential.
    ref_strengths: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0, 4.0])
    # Misspecification scan (formal result, upgraded from the E2-AB ablation cell):
    # 0.0 => correct structured prior; growing => elasticity bounds pushed wrong AND a
    # price-increasing term added until the prior becomes non-monotone (Veblen-like).
    misspec_levels: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0])
    outdir: str = "results"


def smoke() -> "Config":
    c = Config()
    c.sim = SimConfig(horizon=6, n_prices=9, n_ref_bins=15)
    c.data = DataConfig(n_train_traj=60, n_eval_episodes=64,
                        n_segments=3, traj_per_segment=40)
    c.model = ModelConfig(d_model=32, n_layer=1, n_head=2, epochs=3,
                          q_epochs=8, demand_epochs=20, batch_size=64)
    c.exp = ExpConfig(seeds=[0, 1],
                      data_sizes=[60, 200],
                      noise_levels=[0.05, 0.4],
                      suboptimality=[0.1, 0.4],
                      drift_levels=[0.0, 0.5],
                      ref_strengths=[0.0, 1.5, 3.0],
                      misspec_levels=[0.0, 1.0, 3.0])
    return c


def full() -> "Config":
    return Config()


@dataclass(slots=True)
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
