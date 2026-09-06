"""PAXG Forecast Lab Phase P6: Autonomous Tuning, Bayesian Optimization, and Gatekeeper."""

from .bootstrap import BlockBootstrapResult, compute_block_bootstrap_ci, get_block_size_for_timeframe
from .gatekeeper import GatekeeperDecision, LoRAGatekeeper
from .optimizer import EarlyStoppingStagnationCallback, OptunaTPEOptimizer
from .protocol import AutonomousTuningProtocol, MultiSeedEvalSummary, P6RunResult
from .space import (
    ALLOWED_CONTEXTS,
    ALLOWED_FEATURE_SETS,
    ALLOWED_HISTORY_DAYS,
    ALLOWED_RANKS,
    suggest_trial_spec,
    validate_train_spec_invariants,
)

__all__ = [
    "ALLOWED_CONTEXTS",
    "ALLOWED_FEATURE_SETS",
    "ALLOWED_HISTORY_DAYS",
    "ALLOWED_RANKS",
    "AutonomousTuningProtocol",
    "BlockBootstrapResult",
    "EarlyStoppingStagnationCallback",
    "GatekeeperDecision",
    "LoRAGatekeeper",
    "MultiSeedEvalSummary",
    "OptunaTPEOptimizer",
    "P6RunResult",
    "compute_block_bootstrap_ci",
    "get_block_size_for_timeframe",
    "suggest_trial_spec",
    "validate_train_spec_invariants",
]
