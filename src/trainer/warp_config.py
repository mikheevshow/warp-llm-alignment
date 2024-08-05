"""
WARP Configuration class
"""

from dataclasses import dataclass
from utils import OnPolicyConfig

@dataclass
class WARPConfig(OnPolicyConfig):
    """
    WARP Trainer configuration class
    """
    iterations: int = 2
    rl_runs: int = 2
    """amount of rl optimization loops"""
    training_steps: int = 2
    ema_update_rate: int | float = 0.01
    liti_update_rate: int | float = 0.5
    kl_penalty: int | float = 0.5
    """Kulbak-Leibner divergence penalty multiplier"""
