"""Out-of-fold, directed attention summaries from selected checkpoints."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .graph import prepare_graphs
from .splits import validate_splits
from .training import load_checkpoint, write_json


@torch.no_grad()
def export_attention(dataset, run, output, device="auto", prevalence=0.5):
    if not 0 < prevalence <= 1:
        raise ValueError("prevalence must be in (0, 1]")
    dataset.require_labels()
    run, output = Path(run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    payload = json.loads((run / "splits.json").read_text())
    if payload.get("fingerprint") != dataset.fingerprint():
        raise ValueError(
            "Attention export requires the original data and saved split assignments"
        )
    folds = validate_splits(payload["folds"], dataset, len(payload["folds"]))
    model, config, checkpoint, resolved_device = load_checkpoint(
        run / "fold_0" / "best.pt", dataset, device
    )
    graphs = prepare_graphs(dataset, config)
    n = len(dataset.coords)
    count = np.zeros((2, n, n), dtype=np.int64)
    attention_sum = np.zeros((2, n, n), dtype=np.float64)
    phi_sum = np.zeros((n, n, 2), dtype=np.float64)
    for fold_index, split in enumerate(folds):
        if fold_index:
            del model
            model, fold_config, checkpoint, resolved_device = load_checkpoint(
                run / f"fold_{fold_index}" / "best.pt", dataset, device
            )
        else:
            fold_config = config
        if (
            fold_config.to_dict() != config.to_dict()
            or checkpoint["fingerprint"] != payload["fingerprint"]
        ):
            raise ValueError("Fold checkpoint configuration/data do not match this run")
        if (
            checkpoint["test_subject_ids"]
            != dataset.subject_ids[split["test"]].tolist()
        ):
            raise ValueError("Checkpoint test subjects do not match saved splits")
        for index in split["test"]:
            data = graphs[index].data(config.graph.measures).to(resolved_device)
            alpha = (
                model(data, return_attention=True)["attention"].mean(-1).cpu().numpy()
            )
            sender, receiver = data.edge_index.cpu().numpy()
            label = dataset.labels[index]
            count[label, receiver, sender] += 1
            attention_sum[label, receiver, sender] += alpha
            phi_sum[receiver, sender] += data.phi.cpu().numpy()
    rows = []
    for receiver in range(n):
        for sender in range(n):
            counts = count[:, receiver, sender]
            present = int(counts.sum())
            if present < np.ceil(prevalence * len(dataset)) or not np.all(counts > 0):
                continue
            means = attention_sum[:, receiver, sender] / counts
            phi = phi_sum[receiver, sender] / present
            rows.append(
                {
                    "receiver": receiver,
                    "sender": sender,
                    "phi_x": float(phi[0]),
                    "phi_y": float(phi[1]),
                    "attention_0": float(means[0]),
                    "attention_1": float(means[1]),
                    "delta_attention": float(means[1] - means[0]),
                    "prevalence": present / len(dataset),
                    "present_0": int(counts[0]),
                    "present_1": int(counts[1]),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "receiver",
        "sender",
        "phi_x",
        "phi_y",
        "attention_0",
        "attention_1",
        "delta_attention",
        "prevalence",
        "present_0",
        "present_1",
    ]
    with open(output / "edges.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "metadata.json",
        {
            "subjects": len(dataset),
            "edges": len(rows),
            "measures": config.graph.measures,
            "minimum_prevalence": prevalence,
            "direction": "receiver <- sender",
            "aggregation": "last-layer mean over heads; conditional on edge presence",
            "delta": "class 1 minus class 0",
            "partition": "outer test only",
        },
    )
    if rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = np.array([row["delta_attention"] for row in rows])
        limit = max(float(abs(values).max()), 1e-8)
        figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        scatter = axis.scatter(
            [row["phi_x"] for row in rows],
            [row["phi_y"] for row in rows],
            c=values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            s=10,
            alpha=0.8,
        )
        axis.set(
            xlabel=f"log normalized {config.graph.measures[0]}",
            ylabel=f"log normalized {config.graph.measures[1]}",
            title="Out-of-fold directed attention",
        )
        figure.colorbar(scatter, ax=axis, label="Mean attention: class 1 − class 0")
        figure.savefig(output / "ecospace.png", dpi=200)
        plt.close(figure)
    return rows
