"""Minimal CLI. All data locations are supplied by the caller at runtime."""

import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from .config import MEASURES, Config
from .data import load_dataset, make_synthetic
from .training import predict, train_cv, write_json


def parser():
    root = argparse.ArgumentParser(
        description="BGT-ESRE: economy-aligned brain graph classification"
    )
    commands = root.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser(
        "synthetic", help="Generate the single synthetic example cohort"
    )
    synthetic.add_argument("--out", required=True, help="New .npz file")
    synthetic.add_argument("--subjects", type=int, default=80)
    synthetic.add_argument("--regions", type=int, default=32)
    synthetic.add_argument("--timepoints", type=int, default=128)
    synthetic.add_argument("--seed", type=int, default=7)
    train = commands.add_parser(
        "train", help="Train with independent validation and outer test folds"
    )
    train.add_argument("--data", required=True, help="Input .npz file")
    train.add_argument(
        "--config", help="YAML configuration; otherwise use documented defaults"
    )
    train.add_argument("--out", required=True, help="Empty run directory")
    train.add_argument("--splits", help="Previously saved splits.json to reuse")
    train.add_argument("--measures", nargs=2, choices=MEASURES)
    train.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run all ten pairs on identical saved partitions",
    )
    for name in ("epochs", "folds", "seed", "batch-size"):
        train.add_argument(f"--{name}", type=int)
    train.add_argument("--device", default="auto")
    for name in ("predict", "evaluate"):
        command = commands.add_parser(
            name,
            help="Predict unlabeled inputs"
            if name == "predict"
            else "Evaluate a checkpoint",
        )
        command.add_argument("--data", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--device", default="auto")
        if name == "evaluate":
            command.add_argument(
                "--split",
                choices=("test", "all"),
                default="test",
                help="Saved outer test subjects, or an explicitly supplied independent dataset",
            )
    attention = commands.add_parser(
        "attention", help="Export directed attention from held-out subjects only"
    )
    attention.add_argument("--data", required=True)
    attention.add_argument("--run", required=True)
    attention.add_argument("--out", required=True)
    attention.add_argument("--prevalence", type=float, default=0.5)
    attention.add_argument("--device", default="auto")
    smoke = commands.add_parser(
        "smoke",
        help="Synthetic training, checkpoint evaluation, prediction, and attention",
    )
    smoke.add_argument("--out", required=True, help="New empty directory")
    return root


def train(args):
    from .graph import prepare_graphs
    from .splits import load_or_create_splits

    config = Config.load(args.config)
    for name in ("epochs", "folds", "seed", "batch_size"):
        if getattr(args, name) is not None:
            setattr(config.training, name, getattr(args, name))
    if args.measures:
        config.graph.measures = tuple(args.measures)
    config.validate()
    dataset = load_dataset(args.data)
    if not args.all_pairs:
        return train_cv(dataset, config, args.out, args.device, args.splits)
    if args.measures:
        raise ValueError("Choose --measures or --all-pairs")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    load_or_create_splits(dataset, config.training, output, args.splits)
    print("Precomputing all five metrics once per subject...", flush=True)
    prepared = prepare_graphs(dataset, config, MEASURES)
    summaries = {}
    for pair in combinations(MEASURES, 2):
        selected = deepcopy(config)
        selected.graph.measures = pair
        name = "_".join(pair)
        summaries[name] = train_cv(
            dataset,
            selected,
            output / name,
            args.device,
            output / "splits.json",
            prepared,
        )["mean_std"]
    write_json(output / "pairs.json", summaries)
    return summaries


def smoke(output):
    import torch

    from .attention import export_attention

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = make_synthetic(
            root / "synthetic.npz", subjects=30, regions=12, timepoints=40
        )
        config = Config()
        config.model.hidden_dim, config.model.num_heads, config.model.num_layers = (
            8,
            2,
            1,
        )
        config.training.folds, config.training.epochs, config.training.batch_size = (
            3,
            2,
            8,
        )
        train_cv(dataset, config, root / "training", device="cpu")
        checkpoint = root / "training" / "fold_0" / "best.pt"
        predict(dataset, checkpoint, root / "evaluation", "cpu", evaluate=True)
        dataset.labels = None
        predict(dataset, checkpoint, root / "prediction", "cpu")
        export_attention(
            load_dataset(root / "synthetic.npz"),
            root / "training",
            root / "attention",
            "cpu",
        )
    finally:
        torch.set_num_threads(old_threads)
    print(f"End-to-end smoke check passed. Outputs: {root}", flush=True)


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            if Path(args.out).suffix != ".npz":
                raise ValueError("Synthetic output must end in .npz")
            make_synthetic(
                args.out, args.subjects, args.regions, args.timepoints, args.seed
            )
            print(f"Synthetic example saved to {args.out}")
        elif args.command == "train":
            train(args)
        elif args.command in ("predict", "evaluate"):
            result = predict(
                load_dataset(args.data),
                args.checkpoint,
                args.out,
                args.device,
                evaluate=args.command == "evaluate",
                split=getattr(args, "split", "all"),
            )
            if args.command == "evaluate":
                print(result)
        elif args.command == "attention":
            from .attention import export_attention

            export_attention(
                load_dataset(args.data),
                args.run,
                args.out,
                args.device,
                args.prevalence,
            )
        elif args.command == "smoke":
            smoke(args.out)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        command_parser.exit(2, f"Error: {error}\n")
