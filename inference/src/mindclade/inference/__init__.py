"""Mindclade's deterministic model-inference runtime."""

from pkgutil import extend_path

# This regular package exposes the runtime's public API while extending its
# search path to the generated `mindclade.inference.v1alpha1` protocol package.
__path__ = extend_path(__path__, __name__)

from .contracts.adaptive_compute_contract import AdaptiveComputeRequest
from .contracts.execution_mode_contract import ExecutionMode, ExecutionModeRequest
from .contracts.request_contract import InferenceRequest
from .contracts.result_contract import InferenceCandidate, InferenceResult

__all__ = [
    "AdaptiveComputeRequest",
    "ExecutionMode",
    "ExecutionModeRequest",
    "InferenceCandidate",
    "InferenceRequest",
    "InferenceResult",
]

__version__ = "0.1.0"
