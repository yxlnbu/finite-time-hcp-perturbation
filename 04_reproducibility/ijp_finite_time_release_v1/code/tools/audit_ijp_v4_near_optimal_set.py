"""Audit set-valued near-optimal direction--wavenumber selection on V4 data."""

from __future__ import annotations

import csv
import json
from math import cos, pi, sin
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from tools.build_ijp_operator_consistency_v2 import (  # noqa: E402
    log_gain,
    norm_variants,
    rescale_propagator,
    selectors,
)
from tools.build_ijp_reference_reoptimization_v4 import (  # noqa: E402
    REFERENCE_FACTOR,
    REFERENCE_SUBSTEPS,
    RESULT as V4_RESULT,
)
from tools.build_ijp_revision_evidence_v3 import (  # noqa: E402
    K_BOUNDS,
    RESULT as V3_RESULT,
    build_context,
    clean_json,
    coordinates_from_direction,
    direction_angle_degrees,
    direction_from_coordinates,
    evaluate_pool,
    qmc_coordinates,
    registered_anchors,
    require,
)


RESULT = ROOT / "05_results/ijp_v4_near_optimal_set_audit_v1.json"
TABLE = ROOT / "05_results/ijp_v4_near_optimal_set_audit_v1.csv"
EPSILONS = (0.02, 0.01, 0.005, 0.001)
RING_RADII_DEG = (0.5, 1.0, 2.0, 4.0)
RING_AZIMUTHS = 8
K_FACTORS = (0.9, 1.0, 1.1)
K_PROFILE_COUNT = 61
INDEPENDENT_QMC_COUNT = 128


def unique_coordinates(values: list[np.ndarray]) -> np.ndarray:
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for value in values:
        raw = np.asarray(value, dtype=float)
        unique[tuple(np.round(raw, 10))] = raw
    return np.asarray(list(unique.values()), dtype=float)


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = np.cross(n, reference)
    first /= np.linalg.norm(first)
    second = np.cross(n, first)
    second /= np.linalg.norm(second)
    return first, second


def profile_coordinates(centers: list[np.ndarray]) -> np.ndarray:
    values = [*list(qmc_coordinates(INDEPENDENT_QMC_COUNT, 20260831))]
    # Retain every declared centre explicitly.  Direction rings challenge the
    # local basin but do not, by themselves, contain their zero-radius centre;
    # omitting it can make a tight sampled near-optimal set spuriously empty.
    values.extend(np.asarray(center, dtype=float) for center in centers)
    k_profile = np.geomspace(K_BOUNDS[0], K_BOUNDS[1], K_PROFILE_COUNT)
    for center in centers:
        normal = direction_from_coordinates(center)
        center_k = float(np.exp(center[2]))
        for wavenumber in k_profile:
            values.append(coordinates_from_direction(normal, float(wavenumber)))
        first, second = tangent_basis(normal)
        for radius_deg in RING_RADII_DEG:
            radius = np.deg2rad(radius_deg)
            for azimuth in np.linspace(0.0, 2.0 * pi, RING_AZIMUTHS, endpoint=False):
                direction = (
                    cos(radius) * normal
                    + sin(radius) * (cos(azimuth) * first + sin(azimuth) * second)
                )
                direction /= np.linalg.norm(direction)
                for factor in K_FACTORS:
                    values.append(
                        coordinates_from_direction(
                            direction,
                            float(np.clip(center_k * factor, *K_BOUNDS)),
                        )
                    )
    return unique_coordinates(values)


def main() -> int:
    started = time.perf_counter()
    require(V4_RESULT.is_file(), "complete V4 reference re-optimization is required")
    v4 = json.loads(V4_RESULT.read_text(encoding="utf-8"))
    require(
        v4.get("status") == "REFERENCE_REOPTIMIZATION_PASS",
        "V4 reference re-optimization gates are not closed",
    )
    v3 = json.loads(V3_RESULT.read_text(encoding="utf-8"))
    context = build_context("v4_near_optimal_set", factor=REFERENCE_FACTOR)
    output_indices, input_indices = selectors()["constitutive_to_constitutive"]
    active_scales = norm_variants(context.scales)["baseline"]
    reference_indices = np.arange(len(context.shears), dtype=int)
    onset_index = int(np.argmin(np.abs(context.times_s - 1.46e-6)))
    horizon_indices = {
        "onset": reference_indices[: onset_index + 1],
        "terminal": reference_indices,
    }
    anchors = registered_anchors()
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for horizon_id in ("onset", "terminal"):
        key = f"{horizon_id}__baseline__constitutive_to_constitutive"
        winner = v4["records"][key]
        old = v3["reoptimization"]["records"][key]
        centers = [
            coordinates_from_direction(winner["direction_n"], winner["k_m_inv"]),
            coordinates_from_direction(old["direction_n"], old["k_m_inv"]),
        ]
        centers.extend(anchors[:2] if horizon_id == "onset" else anchors[2:3])
        coordinates = profile_coordinates(centers)
        print(
            json.dumps(
                {
                    "stage": "near_optimal_pool",
                    "horizon": horizon_id,
                    "count": int(len(coordinates)),
                }
            ),
            flush=True,
        )
        propagators = evaluate_pool(
            context,
            horizon_indices[horizon_id],
            coordinates,
            substeps=REFERENCE_SUBSTEPS,
        )
        scores = np.asarray(
            [
                log_gain(
                    rescale_propagator(phi, context.scales, active_scales),
                    output_indices,
                    input_indices,
                )
                for phi in propagators
            ]
        )
        winner_coordinates = coordinates_from_direction(
            winner["direction_n"], winner["k_m_inv"]
        )
        gamma_log = max(float(winner["reference_log_gain"]), float(np.max(scores)))
        horizon_sets: dict[str, Any] = {}
        for epsilon in EPSILONS:
            threshold = gamma_log + float(np.log1p(-epsilon))
            selected = np.where(scores >= threshold)[0]
            selected_k = np.exp(coordinates[selected, 2])
            selected_angles = np.asarray(
                [
                    direction_angle_degrees(
                        direction_from_coordinates(coordinates[index]),
                        direction_from_coordinates(winner_coordinates),
                    )
                    for index in selected
                ]
            )
            horizon_sets[f"epsilon_{epsilon:g}"] = {
                "epsilon": epsilon,
                "threshold_log_gain": threshold,
                "sampled_member_count": int(selected.size),
                "k_min_m_inv": None if not selected.size else float(np.min(selected_k)),
                "k_max_m_inv": None if not selected.size else float(np.max(selected_k)),
                "maximum_projective_angle_from_winner_deg": (
                    None if not selected.size else float(np.max(selected_angles))
                ),
            }
        for index, (coordinate, score) in enumerate(
            zip(coordinates, scores, strict=True)
        ):
            normal = direction_from_coordinates(coordinate)
            rows.append(
                {
                    "horizon_id": horizon_id,
                    "sample_index": index,
                    "n1": normal[0],
                    "n2": normal[1],
                    "n3": normal[2],
                    "k_m_inv": float(np.exp(coordinate[2])),
                    "log_gain": float(score),
                    "relative_to_sampled_max_gain": float(
                        np.exp(score - gamma_log)
                    ),
                }
            )
        summaries[horizon_id] = {
            "declared_winner_direction_n": winner["direction_n"],
            "declared_winner_k_m_inv": winner["k_m_inv"],
            "declared_winner_log_gain": winner["reference_log_gain"],
            "sampled_best_log_gain": float(np.max(scores)),
            "sampled_best_direction_n": direction_from_coordinates(
                coordinates[int(np.argmax(scores))]
            ).tolist(),
            "sampled_best_k_m_inv": float(
                np.exp(coordinates[int(np.argmax(scores)), 2])
            ),
            "sample_count": int(len(coordinates)),
            "near_optimal_sets": horizon_sets,
        }

    gates = {
        "independent_pool_does_not_exceed_declared_winner_by_0p1_percent": all(
            value["sampled_best_log_gain"]
            <= value["declared_winner_log_gain"] + np.log(1.001)
            for value in summaries.values()
        ),
        "epsilon_ladder_reported_at_both_horizons": all(
            len(value["near_optimal_sets"]) == len(EPSILONS)
            for value in summaries.values()
        ),
    }
    report = {
        "schema": "IJP_V4_NEAR_OPTIMAL_SET_AUDIT_V1",
        "status": (
            "NEAR_OPTIMAL_SET_AUDIT_PASS"
            if all(gates.values())
            else "NEAR_OPTIMAL_SET_AUDIT_GATES_OPEN"
        ),
        "scope": (
            "Independent scrambled-Sobol challenge plus structured local direction and "
            "full-domain fixed-direction k profiles for the baseline constitutive map. "
            "This is a sampled near-optimal-set audit, not a continuum global proof."
        ),
        "horizons": summaries,
        "gates": gates,
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "artifacts": {"samples_csv": TABLE.relative_to(ROOT).as_posix()},
    }
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    RESULT.write_text(
        json.dumps(clean_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
