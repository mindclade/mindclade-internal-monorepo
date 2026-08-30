"""Stable v1alpha1 training contracts."""

from .checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointPolicy,
    CheckpointRef,
    IncompleteCheckpointError,
)
from .loss import LossReport, LossTerm, NonFiniteLossError, require_finite_loss
from .objective import Objective, OutputFieldObjective
from .optimization import (
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)
from .parallelism import ParallelismConfig, ParallelismMode
from .precision import PrecisionConfig, PrecisionMode
from .program import TrainingProgram
from .reproducibility import (
    ReproducibilityConfig,
    RNGState,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from .state import DataProgress, RunStatus, TrainerState
from .task import BatchEnvelope, TrainingTask, UnsupportedTaskError

__all__ = [
    "BatchEnvelope",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointPolicy",
    "CheckpointRef",
    "DataProgress",
    "IncompleteCheckpointError",
    "LossReport",
    "LossTerm",
    "NonFiniteLossError",
    "Objective",
    "OptimizerConfig",
    "OutputFieldObjective",
    "ParallelismConfig",
    "ParallelismMode",
    "PrecisionConfig",
    "PrecisionMode",
    "RNGState",
    "ReproducibilityConfig",
    "RunStatus",
    "SchedulerConfig",
    "TrainerState",
    "TrainingProgram",
    "TrainingTask",
    "UnsupportedTaskError",
    "build_optimizer",
    "build_scheduler",
    "capture_rng_state",
    "require_finite_loss",
    "restore_rng_state",
    "seed_everything",
]
