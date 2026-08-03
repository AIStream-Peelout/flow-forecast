"""
Warm-start stage two: supervised pretraining of the GR4 parameter head from catchment contexts.

Regresses the classically calibrated per-basin parameters (stage one,
``calibrate_fleet.py``) from the frozen catchment embeddings, in the head's bounded-sigmoid
fraction space. This produces three artifacts:

1. ``parameter_head_<tag>.pt`` — a :class:`GR4SnowParameterHead` state dict whose final biases
   sit at the fleet-median calibrated parameters and whose weights map embedded basins near
   their own calibration. Loaded into training via ``run_training.py --init-parameter-head``.
2. ``context_init_<tag>.pt`` — inverted (optimized) context vectors for calibrated basins that
   lack a pretrained embedding, so their learnable context rows also start at hydrologically
   fitted parameters. Loaded via ``--init-contexts``.
3. ``probe_report_<tag>.json`` — cross-validated held-out R-squared per parameter: the first
   quantitative measure of how much basin-specific hydrology the embeddings encode (the
   "embedding probe"), plus a derived manifest pointing at the new embedding bank.

Contexts are L2-normalized throughout, matching ``--normalize-context`` training.
"""
import argparse
import copy
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from flood_forecast.ode.physics.gr4_calibration import PARAMETER_NAMES
from flood_forecast.ode.physics.hydrology import GR4SnowParameterHead

HEAD_BOUNDS = {
    "X1": (10.0, 2000.0), "X2": (-10.0, 10.0), "X3": (5.0, 500.0), "X4": (2.0, 120.0),
    "Df": (0.0, 0.5), "Tmax": (-2.0, 3.0), "delta": (0.0, 4.0),
}
HEAD_ORDER = ("X1", "X2", "X3", "X4", "Df", "Tmax", "delta")


def targets_to_fractions(calibrations: Dict[str, Dict], site_ids: List[str]) -> torch.Tensor:
    """
    Converts calibrated physical parameters into the head's bounded (0, 1) fraction space.

    :param calibrations: Site -> calibration entry mapping with a ``parameters`` dict.
    :type calibrations: Dict[str, Dict]
    :param site_ids: Ordered sites to convert.
    :type site_ids: List[str]
    :return: Fractions of shape (n_sites, 7) in head order (X1..X4, Df, Tmax, delta), clamped
        away from 0/1 so logit-space targets stay finite.
    :rtype: torch.Tensor
    """
    rows = []
    for site in site_ids:
        params = calibrations[site]["parameters"]
        values = dict(params)
        values["delta"] = params["Tmax"] - params["Tmin"]
        fractions = [(values[name] - HEAD_BOUNDS[name][0]) /
                     (HEAD_BOUNDS[name][1] - HEAD_BOUNDS[name][0]) for name in HEAD_ORDER]
        rows.append(fractions)
    return torch.tensor(rows, dtype=torch.float32).clamp(0.02, 0.98)


def head_fractions(head: GR4SnowParameterHead, contexts: torch.Tensor) -> torch.Tensor:
    """
    Runs the head and converts its physical outputs back to bounded fractions.

    :param head: The parameter head.
    :type head: GR4SnowParameterHead
    :param contexts: L2-normalized contexts of shape (n, embedding_dim).
    :type contexts: torch.Tensor
    :return: Fractions of shape (n, 7) in head order.
    :rtype: torch.Tensor
    """
    emitted = head(contexts)  # (n, 7): X1..X4, Df, Tmax, Tmin
    delta = emitted[:, 5] - emitted[:, 6]
    values = torch.cat([emitted[:, :5], emitted[:, 5].unsqueeze(1), delta.unsqueeze(1)], dim=1)
    lower = torch.tensor([HEAD_BOUNDS[n][0] for n in HEAD_ORDER])
    upper = torch.tensor([HEAD_BOUNDS[n][1] for n in HEAD_ORDER])
    return (values - lower) / (upper - lower)


def make_head(median_fractions: torch.Tensor, embedding_dim: int = 256) -> GR4SnowParameterHead:
    """
    Builds a head whose final biases emit the fleet-median calibrated parameters.

    :param median_fractions: Median target fractions of shape (7,) in head order.
    :type median_fractions: torch.Tensor
    :param embedding_dim: Context width, defaults to 256.
    :type embedding_dim: int, optional
    :return: The bias-initialized head.
    :rtype: GR4SnowParameterHead
    """
    head = GR4SnowParameterHead(embedding_dim=embedding_dim)
    logits = torch.log(median_fractions / (1.0 - median_fractions))
    with torch.no_grad():
        head.net[-1].bias.copy_(logits[:4])
        head.snow_net.bias.copy_(logits[4:])
    return head


def fit_head(head: GR4SnowParameterHead, contexts: torch.Tensor, targets: torch.Tensor,
             epochs: int = 1500, lr: float = 1e-3, weight_decay: float = 1e-3) -> float:
    """
    Fits the head to calibrated fraction targets with full-batch Adam.

    :param head: The head to train in place.
    :type head: GR4SnowParameterHead
    :param contexts: L2-normalized contexts of shape (n, embedding_dim).
    :type contexts: torch.Tensor
    :param targets: Target fractions of shape (n, 7).
    :type targets: torch.Tensor
    :param epochs: Training epochs, defaults to 1500.
    :type epochs: int, optional
    :param lr: Learning rate, defaults to 1e-3.
    :type lr: float, optional
    :param weight_decay: L2 regularization, defaults to 1e-3.
    :type weight_decay: float, optional
    :return: The final training MSE in fraction space.
    :rtype: float
    """
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(head_fractions(head, contexts), targets)
        loss.backward()
        optimizer.step()
    return float(loss)


def cross_validated_probe(contexts: torch.Tensor, targets: torch.Tensor,
                          median_fractions: torch.Tensor, n_folds: int = 5,
                          seed: int = 42) -> Dict[str, float]:
    """
    Reports held-out R-squared per parameter: the embedding probe.

    :param contexts: L2-normalized contexts of shape (n, embedding_dim).
    :type contexts: torch.Tensor
    :param targets: Target fractions of shape (n, 7).
    :type targets: torch.Tensor
    :param median_fractions: Bias-initialization fractions of shape (7,).
    :type median_fractions: torch.Tensor
    :param n_folds: Cross-validation folds, defaults to 5.
    :type n_folds: int, optional
    :param seed: Fold-assignment seed, defaults to 42.
    :type seed: int, optional
    :return: Parameter name -> held-out R-squared.
    :rtype: Dict[str, float]
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(contexts))
    folds = np.array_split(order, n_folds)
    predictions = torch.zeros_like(targets)
    for fold in folds:
        train_index = torch.tensor(np.setdiff1d(order, fold), dtype=torch.long)
        head = make_head(median_fractions, embedding_dim=contexts.shape[1])
        fit_head(head, contexts[train_index], targets[train_index])
        with torch.no_grad():
            predictions[fold] = head_fractions(head, contexts[torch.tensor(fold)])
    report = {}
    for column, name in enumerate(HEAD_ORDER):
        residual = ((predictions[:, column] - targets[:, column]) ** 2).sum()
        variance = ((targets[:, column] - targets[:, column].mean()) ** 2).sum()
        report[name] = round(float(1.0 - residual / variance.clamp(min=1e-12)), 3)
    return report


def invert_contexts(head: GR4SnowParameterHead, targets: torch.Tensor, embedding_dim: int,
                    steps: int = 800, lr: float = 1e-2, seed: int = 42) -> torch.Tensor:
    """
    Optimizes unit-norm context vectors that reproduce given targets through a frozen head.

    :param head: The pretrained (frozen) head.
    :type head: GR4SnowParameterHead
    :param targets: Target fractions of shape (n, 7).
    :type targets: torch.Tensor
    :param embedding_dim: Context width.
    :type embedding_dim: int
    :param steps: Optimization steps, defaults to 800.
    :type steps: int, optional
    :param lr: Learning rate, defaults to 1e-2.
    :type lr: float, optional
    :param seed: Initialization seed, defaults to 42.
    :type seed: int, optional
    :return: L2-normalized contexts of shape (n, embedding_dim).
    :rtype: torch.Tensor
    """
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(len(targets), embedding_dim, generator=generator, requires_grad=True)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        contexts = torch.nn.functional.normalize(raw, dim=-1)
        loss = torch.nn.functional.mse_loss(head_fractions(head, contexts), targets)
        loss.backward()
        optimizer.step()
    return torch.nn.functional.normalize(raw.detach(), dim=-1)


def main() -> None:
    """
    CLI entry point for stage-two head pretraining.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(description="Pretrain the GR4 parameter head")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--embedding-bank", required=True,
                        help="Path to embeddings_<fusion>.pt (e.g. COUT_v2)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="coutv2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.calibration) as f:
        calibrations = {s: v for s, v in json.load(f).items() if v.get("status") == "ok"}
    bank = torch.load(args.embedding_bank, weights_only=False)
    bank_lookup = {site: i for i, site in enumerate(bank["site_ids"])}
    embedding_dim = bank["embeddings"].shape[1]

    non_holdout = [b["site_id"] for b in manifest["basins"] if b.get("split") != "holdout"]
    embedded = [s for s in non_holdout if s in calibrations and s in bank_lookup]
    unembedded = [s for s in non_holdout if s in calibrations and s not in bank_lookup]
    contexts = torch.nn.functional.normalize(
        torch.stack([bank["embeddings"][bank_lookup[s]] for s in embedded]), dim=-1)
    targets = targets_to_fractions(calibrations, embedded)
    median_fractions = targets.median(dim=0).values
    print("regression set: %d embedded basins; inversion set: %d" %
          (len(embedded), len(unembedded)))

    probe = cross_validated_probe(contexts, targets, median_fractions, seed=args.seed)
    print("embedding probe (held-out R^2 per parameter):", probe)

    head = make_head(median_fractions, embedding_dim)
    final_mse = fit_head(head, contexts, targets)
    head_path = os.path.join(args.output_dir, "parameter_head_%s.pt" % args.tag)
    torch.save(head.state_dict(), head_path)

    inverted = invert_contexts(head, targets_to_fractions(calibrations, unembedded),
                               embedding_dim, seed=args.seed) if unembedded else torch.empty(0)
    context_path = os.path.join(args.output_dir, "context_init_%s.pt" % args.tag)
    torch.save({"site_ids": unembedded, "contexts": inverted}, context_path)

    derived = copy.deepcopy(manifest)
    derived["embedding_path"] = os.path.abspath(args.embedding_bank)
    for basin in derived["basins"]:
        basin["has_embedding"] = basin["site_id"] in bank_lookup
    derived_path = os.path.join(args.output_dir, "manifest_%s.json" % args.tag)
    with open(derived_path, "w") as f:
        json.dump(derived, f, indent=1)

    report = {
        "probe_r2": probe, "final_fit_mse": round(final_mse, 5),
        "n_embedded": len(embedded), "n_inverted": len(unembedded),
        "median_parameters": {n: round(float(
            HEAD_BOUNDS[n][0] + median_fractions[i] * (HEAD_BOUNDS[n][1] - HEAD_BOUNDS[n][0])),
            3) for i, n in enumerate(HEAD_ORDER)},
        "artifacts": {"head": head_path, "context_init": context_path,
                      "manifest": derived_path},
    }
    with open(os.path.join(args.output_dir, "probe_report_%s.json" % args.tag), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
