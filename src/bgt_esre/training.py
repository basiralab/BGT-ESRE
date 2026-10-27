"""Training, checkpoint-bound inference, and held-out evaluation."""

import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from . import __version__
from .config import Config
from .graph import prepare_graphs
from .model import BGTESRE
from .splits import load_or_create_splits


def write_json(path, value):
    """Write portable JSON: undefined metrics are null, never NaN/Infinity."""

    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities[:, 1]))
        if len(set(labels)) == 2
        else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def make_loader(graphs, indices, pair, batch_size, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        [graphs[i].data(pair) for i in indices],
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    logits = []
    for batch in loader:
        logits.append(model(batch.to(device))["logits"].cpu())
    logits = torch.cat(logits)
    return logits.numpy(), logits.softmax(-1).numpy()


def save_predictions(path, dataset, indices, probabilities):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["subject_id", "label", "prediction", "probability_0", "probability_1"]
        )
        for index, probs in zip(indices, probabilities):
            label = "" if dataset.labels is None else int(dataset.labels[index])
            writer.writerow(
                [
                    dataset.subject_ids[index],
                    label,
                    int(probs.argmax()),
                    float(probs[0]),
                    float(probs[1]),
                ]
            )


def runtime_metadata(device):
    source_root = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    try:
        # Record the source checkout only. A wheel must not claim the Git
        # revision of an unrelated repository used as the caller's directory.
        if (source_root / ".git").exists():
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=source_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=source_root, text=True
                )
            )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "package_version": __version__,
        "device": str(device),
        "accelerator": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "git_revision": revision,
        "git_dirty": dirty,
    }


def train_cv(dataset, config, output, device="auto", split_file=None, prepared=None):
    config.validate()
    dataset.require_labels()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    folds = load_or_create_splits(dataset, config.training, output, split_file)
    write_json(output / "config.json", config.to_dict())
    device = resolve_device(device)
    write_json(output / "environment.json", runtime_metadata(device))
    print("Preparing subject graphs and fixed ecospace coordinates...", flush=True)
    graphs = prepared if prepared is not None else prepare_graphs(dataset, config)
    fingerprint = dataset.fingerprint()
    results = []
    oof_probabilities = np.empty((len(dataset), 2), dtype=np.float32)
    c = config.training
    for fold_index, split in enumerate(folds):
        seed = c.seed + fold_index
        seed_everything(seed)
        directory = output / f"fold_{fold_index}"
        directory.mkdir()
        model = BGTESRE(config, dataset.coords.shape[0]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=c.scheduler_factor,
            patience=c.scheduler_patience,
        )
        train_loader = make_loader(
            graphs, split["train"], config.graph.measures, c.batch_size, True, seed
        )
        validation_loader = make_loader(
            graphs, split["validation"], config.graph.measures, c.batch_size
        )
        best_auc, best_loss, stale = -float("inf"), float("inf"), 0
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, c.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(batch)["logits"], batch.y)
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Non-finite training loss; inspect data scale and optimization settings"
                    )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * batch.num_graphs
            logits, probabilities = infer(model, validation_loader, device)
            labels = dataset.labels[split["validation"]]
            validation = metrics(labels, probabilities)
            validation_loss = float(
                F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels))
            )
            if not np.isfinite(validation_loss) or not np.isfinite(validation["auc"]):
                raise ValueError("Validation loss and AUC must be finite")
            record = {
                "epoch": epoch,
                "training_loss": total_loss / len(split["train"]),
                "validation_loss": validation_loss,
                "validation": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_gamma": [
                    float(layer.attention.gamma.detach().tanh())
                    for layer in model.layers
                ],
            }
            with open(directory / "history.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            auc = validation["auc"]
            improved = auc > best_auc or (
                auc == best_auc and validation_loss < best_loss
            )
            scheduler.step(auc)
            if improved:
                best_auc, best_loss, stale = auc, validation_loss, 0
                torch.save(
                    {
                        "format_version": 1,
                        "config": config.to_dict(),
                        "num_regions": model.num_regions,
                        "coords": torch.tensor(dataset.coords),
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "validation_metrics": validation,
                        "fingerprint": fingerprint,
                        "test_subject_ids": dataset.subject_ids[split["test"]].tolist(),
                        "fold": fold_index,
                        "seed": seed,
                    },
                    directory / "best.pt",
                )
            else:
                stale += 1
            print(
                f"Fold {fold_index + 1}/{len(folds)} epoch {epoch}: loss={record['training_loss']:.4f} "
                f"validation AUC={auc:.4f}",
                flush=True,
            )
            if stale >= c.patience:
                break
        checkpoint = torch.load(
            directory / "best.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state"])
        test_loader = make_loader(
            graphs, split["test"], config.graph.measures, c.batch_size
        )
        _, probabilities = infer(model, test_loader, device)
        # The outer test set is first evaluated after checkpoint selection.
        oof_probabilities[split["test"]] = probabilities
        result = {
            "fold": fold_index,
            "selected_epoch": checkpoint["epoch"],
            "validation": checkpoint["validation_metrics"],
            "test": metrics(dataset.labels[split["test"]], probabilities),
            "training_seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
            "partition_sizes": {key: len(value) for key, value in split.items()},
        }
        if device.type == "cuda":
            result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
        write_json(directory / "metrics.json", result)
        save_predictions(
            directory / "predictions.csv", dataset, split["test"], probabilities
        )
        results.append(result)
        # Release the old optimizer's parameter/state references before the
        # next fold allocates a fresh model (important for large ROI encoders).
        del model, optimizer, scheduler, checkpoint
    summary = {
        "folds": results,
        "mean_std": {},
        "pooled_out_of_fold": metrics(dataset.labels, oof_probabilities),
        "standard_deviation": "sample standard deviation across outer test folds (ddof=1)",
    }
    for name in ("accuracy", "auc", "sensitivity", "specificity"):
        values = [fold["test"][name] for fold in results]
        summary["mean_std"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    write_json(output / "metrics.json", summary)
    save_predictions(
        output / "predictions.csv",
        dataset,
        list(range(len(dataset))),
        oof_probabilities,
    )
    print(f"Test-fold mean AUC: {summary['mean_std']['auc']['mean']:.4f}", flush=True)
    return summary


def load_checkpoint(path, dataset, device="auto"):
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if dataset.coords.shape != tuple(checkpoint["coords"].shape) or not np.allclose(
        dataset.coords, checkpoint["coords"].numpy(), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("Input atlas coordinates/order do not match the checkpoint")
    config = Config.from_dict(checkpoint["config"])
    model = BGTESRE(config, checkpoint["num_regions"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint, device


def predict(
    dataset, checkpoint_path, output, device="auto", evaluate=False, split="test"
):
    model, config, checkpoint, device = load_checkpoint(
        checkpoint_path, dataset, device
    )
    if evaluate:
        dataset.require_labels()
    if evaluate and split == "test":
        if dataset.fingerprint() != checkpoint["fingerprint"]:
            raise ValueError(
                "Test evaluation requires the original input data; use --split all for independent labeled data"
            )
        mapping = {
            identifier: index for index, identifier in enumerate(dataset.subject_ids)
        }
        indices = [mapping[identifier] for identifier in checkpoint["test_subject_ids"]]
    else:
        indices = list(range(len(dataset)))
    graphs = prepare_graphs(dataset, config)
    loader = make_loader(
        graphs, indices, config.graph.measures, config.training.batch_size
    )
    _, probabilities = infer(model, loader, device)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(output / "predictions.csv", dataset, indices, probabilities)
    if evaluate:
        result = {
            "split": split,
            "subjects": len(indices),
            "metrics": metrics(dataset.labels[indices], probabilities),
        }
        write_json(output / "metrics.json", result)
        return result
    return probabilities
