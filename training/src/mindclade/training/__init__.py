"""Mindclade's reproducible PyTorch training contracts and reference engine."""

from importlib.metadata import PackageNotFoundError, version

from .api import (
    BatchEnvelope,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    CheckpointPolicy,
    CheckpointRef,
    DataProgress,
    LossReport,
    LossTerm,
    Objective,
    OptimizerConfig,
    OutputFieldObjective,
    ParallelismConfig,
    ParallelismMode,
    PrecisionConfig,
    PrecisionMode,
    ReproducibilityConfig,
    SchedulerConfig,
    TrainerState,
    TrainingProgram,
    TrainingTask,
)
from .checkpointing import DCPCheckpointManager, RestoredCheckpoint
from .core import (
    DeterministicSyntheticDataset,
    SyntheticSplit,
    Trainer,
    TrainingResult,
    build_synthetic_loader,
)
from .tasks import MultitaskDiffusionTask

try:
    __version__ = version("mindclade-training")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "BatchEnvelope",
    "CheckpointCompatibilityError",
    "CheckpointIntegrityError",
    "CheckpointPolicy",
    "CheckpointRef",
    "DCPCheckpointManager",
    "DataProgress",
    "DeterministicSyntheticDataset",
    "LossReport",
    "LossTerm",
    "MultitaskDiffusionTask",
    "Objective",
    "OptimizerConfig",
    "OutputFieldObjective",
    "ParallelismConfig",
    "ParallelismMode",
    "PrecisionConfig",
    "PrecisionMode",
    "ReproducibilityConfig",
    "RestoredCheckpoint",
    "SchedulerConfig",
    "SyntheticSplit",
    "Trainer",
    "TrainerState",
    "TrainingProgram",
    "TrainingResult",
    "TrainingTask",
    "build_synthetic_loader",
]
