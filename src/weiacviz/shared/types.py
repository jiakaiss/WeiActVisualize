"""Shared data types used across modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Granularity(str, Enum):
    """Quantization granularity."""

    PER_TENSOR = "per-tensor"
    PER_CHANNEL = "per-channel"
    PER_GROUP = "per-group"


class Symmetry(str, Enum):
    """Quantization symmetry mode."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class StatLevel(str, Enum):
    """Statistic aggregation level."""

    PER_TENSOR = "per-tensor"
    PER_CHANNEL = "per-channel"
    PER_GROUP = "per-group"


class ModuleKind(str, Enum):
    """Functional kind of a module within the model."""

    ATTENTION = "attn"
    MLP = "mlp"
    OTHER = "other"


class TensorRole(str, Enum):
    """Whether a tensor is a weight or an activation."""

    WEIGHT = "weight"
    ACTIVATION = "activation"


@dataclass
class HistogramResult:
    counts: list
    bin_edges: list
    num_bins: int


@dataclass
class StatResult:
    """Aggregated statistics for one tensor at one level."""

    module_path: str
    role: TensorRole
    level: StatLevel
    min: float
    max: float
    mean: float
    std: float
    percentiles: dict = field(default_factory=dict)
    histogram: Optional[HistogramResult] = None
    kurtosis: float = float("nan")
    skewness: float = float("nan")
    tail_ratio: float = float("nan")
    outlier_ratio: float = float("nan")
    shape_label: str = ""


@dataclass
class QuantConfig:
    """Configuration for a fake-quantization scheme."""

    bits: int = 8
    granularity: Granularity = Granularity.PER_TENSOR
    symmetry: Symmetry = Symmetry.SYMMETRIC
    group_size: Optional[int] = None  # required when granularity == PER_GROUP
    quantize_weights: bool = True
    quantize_activations: bool = False

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    @property
    def qmin(self) -> int:
        if self.symmetry == Symmetry.SYMMETRIC:
            return -(1 << (self.bits - 1))
        return 0


@dataclass
class CaptureConfig:
    """Configuration for activation capture during calibration."""

    capture_inputs: bool = True
    capture_outputs: bool = True
    max_samples: int = 128
    batch_size: int = 8


@dataclass
class ModuleInfo:
    """Metadata describing a target module."""

    path: str
    kind: ModuleKind
    shape: tuple
    dtype: str
