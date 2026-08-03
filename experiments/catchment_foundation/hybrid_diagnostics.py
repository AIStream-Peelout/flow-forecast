"""Parameter and context diagnostics for multi-basin hybrid GR4 experiments."""
import json
import os
from typing import Dict, Iterable

import numpy as np
import torch


PARAMETER_NAMES = ("X1", "X2", "X3", "X4", "Df", "Tmax", "delta")


def save_parameter_diagnostics(model, manifest_path: str, out_dir: str,
                               training_positions: Iterable[int]) -> Dict:
    """Save emitted physics parameters, boundary saturation, context norms, and ASOS gates."""
    with open(manifest_path) as file:
        manifest = json.load(file)
    positions = torch.arange(len(manifest["basins"]), device=model.flow_scales.device)
    training_positions = set(int(position) for position in training_positions)
    with torch.no_grad():
        contexts = model.basin_context(positions)
        emitted = model.hybrid.parameter_head(contexts)
        delta = emitted[:, 5] - emitted[:, 6]
        values = torch.cat([emitted[:, :6], delta.unsqueeze(1)], dim=1)
        head = model.hybrid.parameter_head
        lower = torch.cat([head.lower, head.snow_lower]).to(values)
        upper = torch.cat([head.upper, head.snow_upper]).to(values)
        fractions = (values - lower) / (upper - lower).clamp(min=1e-8)
        saturated = (fractions < 0.01) | (fractions > 0.99)
        gates = None
        generator = model.hybrid.forcing_generator
        if generator.anchored and generator.use_asos_gate:
            gates = torch.sigmoid(generator.gate_net(contexts)).squeeze(1)

    rows = []
    for index, basin in enumerate(manifest["basins"]):
        row = {
            "site_id": basin["site_id"],
            "split": basin.get("split"),
            "used_for_training": index in training_positions,
            "has_pretrained_context": bool(model.has_fixed_context[index]),
            "context_norm": float(contexts[index].norm()),
            "parameters": {
                name: float(values[index, column])
                for column, name in enumerate(PARAMETER_NAMES)},
            "saturated": {
                name: bool(saturated[index, column])
                for column, name in enumerate(PARAMETER_NAMES)},
        }
        if gates is not None:
            row["asos_gate"] = float(gates[index])
        rows.append(row)

    groups = {}
    group_masks = {
        "training": np.array([row["used_for_training"] for row in rows]),
        "basin_validation": np.array([row["split"] == "basin_valid" for row in rows]),
        "holdout": np.array([row["split"] == "holdout" for row in rows]),
    }
    saturation_np = saturated.cpu().numpy()
    for name, mask in group_masks.items():
        selected = np.flatnonzero(mask)
        if not len(selected):
            continue
        group = {
            "n_basins": int(len(selected)),
            "all_gr4_parameters_saturated_pct": round(
                100.0 * float(saturation_np[selected, :4].all(axis=1).mean()), 1),
            "any_parameter_saturated_pct": round(
                100.0 * float(saturation_np[selected].any(axis=1).mean()), 1),
        }
        if gates is not None:
            gate_values = gates[selected].cpu().numpy()
            group["asos_gate_min_median_max"] = [
                round(float(np.min(gate_values)), 6),
                round(float(np.median(gate_values)), 6),
                round(float(np.max(gate_values)), 6),
            ]
        groups[name] = group

    result = {"groups": groups, "basins": rows}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "parameter_diagnostics.json"), "w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({"parameter_diagnostics": groups}, indent=2))
    return result
