"""One validated configuration shared by the library, CLI, and checkpoints."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

MEASURES = ("ediff", "erout", "comm", "ecc", "ebc")


@dataclass
class GraphConfig:
    measures: tuple[str, str] = ("ediff", "comm")
    density: float = 0.15
    distance_decay: float | str = 0.1
    weighting: str = "economy"
    eps: float = 1e-8


@dataclass
class ModelConfig:
    input_mode: str = "bold"
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    ffn_multiplier: int = 2
    cnn_kernels: tuple[int, ...] = (7, 5, 3)
    temporal_downsample: int = 4
    standardize_bold: bool = True
    attention: str = "rotary"


@dataclass
class TrainingConfig:
    folds: int = 5
    validation_fraction: float = 0.2
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    patience: int = 15
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    seed: int = 42


@dataclass
class Config:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        g, m, t = self.graph, self.model, self.training
        if (
            len(g.measures) != 2
            or len(set(g.measures)) != 2
            or not set(g.measures) <= set(MEASURES)
        ):
            raise ValueError(
                f"graph.measures must contain two distinct names from {MEASURES}"
            )
        if not 0 < g.density <= 1 or not 0 < g.eps < 1:
            raise ValueError("graph.density must be in (0, 1]; eps must be in (0, 1)")
        if g.weighting not in ("economy", "connectivity"):
            raise ValueError("graph.weighting must be economy or connectivity")
        if g.distance_decay != "inverse_mean_distance":
            if (
                not isinstance(g.distance_decay, (float, int))
                or not math.isfinite(g.distance_decay)
                or g.distance_decay < 0
            ):
                raise ValueError(
                    "distance_decay must be nonnegative or inverse_mean_distance"
                )
        if m.input_mode not in ("bold", "profile") or m.attention not in (
            "rotary",
            "additive",
        ):
            raise ValueError(
                "input_mode must be bold/profile; attention must be rotary/additive"
            )
        if (
            m.hidden_dim < 1
            or m.num_heads < 1
            or m.hidden_dim % m.num_heads
            or (m.hidden_dim // m.num_heads) % 2
        ):
            raise ValueError("hidden_dim / num_heads must be a positive even integer")
        if m.num_layers < 1 or m.ffn_multiplier < 1 or not 0 <= m.dropout < 1:
            raise ValueError(
                "layers and ffn_multiplier must be positive; dropout must be in [0, 1)"
            )
        if (
            m.temporal_downsample < 1
            or not m.cnn_kernels
            or any(k < 1 or k % 2 == 0 for k in m.cnn_kernels)
        ):
            raise ValueError(
                "temporal_downsample must be positive; CNN kernels must be positive odd integers"
            )
        if t.folds < 2 or not 0 < t.validation_fraction < 0.5:
            raise ValueError(
                "folds must be >=2; validation_fraction must be in (0, 0.5)"
            )
        if min(t.epochs, t.batch_size, t.patience, t.scheduler_patience) < 1:
            raise ValueError("epochs, batch_size, and patience values must be positive")
        if (
            not math.isfinite(t.learning_rate)
            or t.learning_rate <= 0
            or not math.isfinite(t.weight_decay)
            or t.weight_decay < 0
        ):
            raise ValueError(
                "learning_rate must be positive; weight_decay must be nonnegative"
            )
        if not 0 < t.scheduler_factor < 1 or t.seed < 0:
            raise ValueError(
                "scheduler_factor must be in (0, 1); seed must be nonnegative"
            )
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {"graph", "model", "training"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        result = cls()
        for name, kind in (
            ("graph", GraphConfig),
            ("model", ModelConfig),
            ("training", TrainingConfig),
        ):
            section = values.get(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be a mapping")
            unknown = set(section) - {f.name for f in fields(kind)}
            if unknown:
                raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
            setattr(result, name, kind(**section))
        return result.validate()

    @classmethod
    def load(cls, path: str | Path | None = None):
        if path is None:
            return cls().validate()
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})
