"""Assemble and verify the publication-ready finite-time reproducibility pack."""

from __future__ import annotations

import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import matplotlib
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "04_reproducibility/ijp_finite_time_release_v1"
ARCHIVE = ROOT / "04_reproducibility/ijp_finite_time_release_v1.zip"
RECEIPT = ROOT / "05_results/ijp_reproducibility_release_v1.json"


DATA_FILES = (
    "05_results/bai_1982_scalar_regression_reference_v5_independent_redteam_v1.json",
    "05_results/ijp_core_figure_set_v1.json",
    "05_results/ijp_operator_consistency_v2.json",
    "05_results/ijp_operator_consistency_v2_arrays.npz",
    "05_results/ijp_operator_consistency_v2/norm_selector_sensitivity.csv",
    "05_results/ijp_operator_consistency_v2/norm_selector_winners.csv",
    "05_results/ijp_operator_consistency_v2/mechanism_history.csv",
    "05_results/ijp_strengthening_evidence_v1.json",
    "05_results/ijp_strengthening_evidence_v1_arrays.npz",
    "05_results/ijp_material_parameter_provenance_v1.json",
    "05_results/ijp_material_parameter_provenance_v1.csv",
    "05_results/ijp_strengthening_evidence_v1/positive_controls.csv",
    "05_results/ijp_strengthening_evidence_v1/orientation_transfer.csv",
    "05_results/ijp_strengthening_evidence_v1/global_search_levels.csv",
    "05_results/ijp_strengthening_evidence_v1/optimizer_trajectories.csv",
    "05_results/cp_ti_dynamic_descriptor_qz_v1.json",
    "05_results/cp_ti_finite_time_dynamic_perturbation_v2.json",
    "05_results/cp_ti_continuous_spectrum_robustness_v2.json",
    "05_results/cp_ti_finite_time_discrimination_v1.json",
    "05_results/cp_ti_mechanism_causality_v2.json",
    "05_results/cp_ti_texture_pathway_v1.json",
    "05_results/hcp_qs_spectrum_verification_seed_v1.json",
    "05_results/cp_ti_fwhm_seed_spectrum_v1.json",
    "05_results/cp_ti_lsbrp_q3_nonautonomous_bridge_v1.json",
    "05_results/ijp_singular_vector_nonlinear_validation_v2.json",
    "05_results/ijp_singular_vector_nonlinear_validation_v2.csv",
    "05_results/ijp_singular_vector_nonlinear_validation_v2_fields.npz",
)

CODE_FILES = (
    "pyproject.toml",
    "tools/build_ijp_operator_consistency_v2.py",
    "tools/build_ijp_strengthening_evidence_v1.py",
    "tools/run_ijp_singular_vector_nonlinear_validation_v2.py",
    "tools/build_ijp_material_parameter_provenance_v1.py",
    "tools/build_ijp_reproducibility_release_v1.py",
    "tools/build_ijp_finite_time_discrimination_v1.py",
    "tools/audit_cp_ti_continuous_spectrum_robustness_v1.py",
    "tools/audit_cp_ti_continuous_spectrum_robustness_v2.py",
    "tools/run_cp_ti_texture_pathway_v1.py",
    "tools/run_cp_ti_finite_time_propagator_v2.py",
    "tools/audit_cp_ti_finite_time_v2_robustness_v1.py",
    "tools/run_cp_ti_lsbrp_periodic69_v1.py",
    "tools/run_cp_ti_lsbrp_q1_dense_history_v1.py",
    "tests/test_ijp_operator_consistency_v2.py",
    "tests/test_ijp_singular_vector_nonlinear_validation_v2.py",
    "src/hcp_cp_gnd/__init__.py",
    "src/hcp_cp_gnd/dynamic_crystal_perturbation_v1.py",
    "src/hcp_cp_gnd/dynamic_mechanism_causality_v1.py",
    "src/hcp_cp_gnd/evolving_work_partition_v1.py",
    "src/hcp_cp_gnd/lsbrp_metrics_v1.py",
    "src/hcp_cp_gnd/nonlinear_mechanism_switches_v1.py",
    "src/hcp_cp_gnd/periodic69_history_v1.py",
    "src/hcp_cp_gnd/periodic69_nonlinear_v1.py",
    "src/hcp_cp_gnd/qs_descriptor.py",
    "src/hcp_cp_gnd/sl3_chart.py",
    "src/hcp_cp_gnd/state_contract.py",
    "src/hcp_cp_gnd/texture_pathway_atlas_v1.py",
    "src/hcp_cp_gnd/cp_ti_material_v1.py",
    "src/hcp_cp_gnd/spectral_export.py",
    "src/hcp_cp_gnd/micromorphic.py",
    "config/cp_ti_grade1_dynamic_hcp_v1.json",
)

MANUSCRIPT_FILES = (
    "06_manuscript/ijp_spectral_hcp/main_finite_time.pdf",
    "06_manuscript/ijp_spectral_hcp/main_finite_time.tex",
    "06_manuscript/ijp_spectral_hcp/sections/ft_introduction.tex",
    "06_manuscript/ijp_spectral_hcp/sections/ft_numerical_methods.tex",
    "06_manuscript/ijp_spectral_hcp/sections/ft_results.tex",
    "06_manuscript/ijp_spectral_hcp/sections/ft_discussion.tex",
    "06_manuscript/ijp_spectral_hcp/sections/ft_conclusions.tex",
    "06_manuscript/ijp_spectral_hcp/sections/ft_model.tex",
    "06_manuscript/ijp_spectral_hcp/appendices/ft_reproducibility.tex",
    "06_manuscript/ijp_spectral_hcp/tables/ft_material_parameter_source_table.tex",
    "06_manuscript/ijp_spectral_hcp/metadata/finite_time_claim_evidence_v2.csv",
    "06_manuscript/ijp_spectral_hcp/figures/fig08_positive_orientation_anchor_audit.pdf",
    "06_manuscript/ijp_spectral_hcp/figures/fig09_operator_consistency_sensitivity.pdf",
    "06_manuscript/ijp_spectral_hcp/figures/fig10_singular_vector_nonlinear_validation.pdf",
    "06_manuscript/ijp_spectral_hcp/IJP_STRENGTHENING_AUDIT_V2.md",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_registered(relative: str, category: str) -> Path:
    source = ROOT / relative
    require(source.is_file(), f"missing release input: {source}")
    target = RELEASE / category / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def command_output(arguments: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "arguments": arguments,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sanitize_runtime_configuration(value: object) -> object:
    """Remove build-machine paths while retaining numerical-library identity."""

    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in ("path", "directory", "command")):
                continue
            cleaned[str(key)] = sanitize_runtime_configuration(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_runtime_configuration(item) for item in value]
    return value


def public_requirements_lock() -> str:
    packages = ("numpy", "scipy", "matplotlib", "pytest")
    lines = ["# Minimal public runtime for the released evidence scripts"]
    lines.extend(f"{name}=={package_version(name)}" for name in packages)
    lines.append("# Local package source is included under code/src/hcp_cp_gnd")
    return "\n".join(lines) + "\n"


def array_contract(path: Path) -> dict[str, object]:
    required = {
        "baseline_onset_x_generators": (129, 69, 69),
        "baseline_onset_y_generators": (129, 69, 69),
        "baseline_terminal_y_generators": (129, 69, 69),
        "baseline_onset_x_input_vector": (69,),
        "baseline_onset_x_output_vector": (69,),
        "baseline_onset_y_input_vector": (69,),
        "baseline_onset_y_output_vector": (69,),
        "baseline_terminal_y_input_vector": (69,),
        "baseline_terminal_y_output_vector": (69,),
        "global_qmc_coordinates": (1024, 3),
        "global_qmc_onset_log_gain": (1024,),
        "global_qmc_terminal_log_gain": (1024,),
        "orientation_minimum_storage_generators": (17, 69, 69),
        "orientation_median_storage_generators": (17, 69, 69),
        "orientation_maximum_storage_generators": (17, 69, 69),
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        shape_failures = {
            key: {"expected": list(shape), "actual": list(archive[key].shape)}
            for key, shape in required.items()
            if key in archive.files and archive[key].shape != shape
        }
        nonfinite = [
            key
            for key in required
            if key in archive.files and not np.all(np.isfinite(archive[key]))
        ]
        complex_generator_keys = [key for key in archive.files if key.endswith("_generators")]
        complex_vector_keys = [
            key
            for key in archive.files
            if key.endswith("_input_vector") or key.endswith("_output_vector")
        ]
        result = {
            "array_count": len(archive.files),
            "required_array_count": len(required),
            "missing_required_arrays": missing,
            "shape_failures": shape_failures,
            "nonfinite_required_arrays": nonfinite,
            "generator_array_keys": complex_generator_keys,
            "singular_vector_array_keys": complex_vector_keys,
            "all_generators_complex128": all(
                archive[key].dtype == np.complex128 for key in complex_generator_keys
            ),
            "all_singular_vectors_complex128": all(
                archive[key].dtype == np.complex128 for key in complex_vector_keys
            ),
        }
    result["contract_pass"] = not missing and not shape_failures and not nonfinite
    return result


def required_npz_contract(
    path: Path, required: dict[str, tuple[int, ...]]
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        shape_failures = {
            key: {"expected": list(shape), "actual": list(archive[key].shape)}
            for key, shape in required.items()
            if key in archive.files and archive[key].shape != shape
        }
        nonfinite = [
            key
            for key in required
            if key in archive.files and not np.all(np.isfinite(archive[key]))
        ]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "missing_required_arrays": missing,
        "shape_failures": shape_failures,
        "nonfinite_required_arrays": nonfinite,
        "contract_pass": not missing and not shape_failures and not nonfinite,
    }


def main() -> int:
    RELEASE.mkdir(parents=True, exist_ok=True)
    obsolete_release_figure = (
        RELEASE
        / "manuscript/06_manuscript/ijp_spectral_hcp/figures/fig08_positive_orientation_global_certification.pdf"
    )
    if obsolete_release_figure.is_file():
        obsolete_release_figure.unlink()
    copied = []
    for relative in DATA_FILES:
        copied.append(copy_registered(relative, "data"))
    for relative in CODE_FILES:
        copied.append(copy_registered(relative, "code"))
    for relative in MANUSCRIPT_FILES:
        copied.append(copy_registered(relative, "manuscript"))

    source_array = ROOT / "05_results/ijp_strengthening_evidence_v1_arrays.npz"
    contract = array_contract(source_array)
    require(bool(contract["contract_pass"]), f"release array contract failed: {contract}")
    operator_contract = required_npz_contract(
        ROOT / "05_results/ijp_operator_consistency_v2_arrays.npz",
        {
            "times_s": (129,),
            "onset_x_propagators": (129, 69, 69),
            "onset_y_propagators": (129, 69, 69),
            "terminal_y_propagators": (129, 69, 69),
            "onset_x_dominant_mode_condition_number": (129,),
            "onset_y_adjacent_leader_rotation_deg": (129,),
            "terminal_y_adjacent_commutator": (129,),
        },
    )
    nonlinear_contract = required_npz_contract(
        ROOT / "05_results/ijp_singular_vector_nonlinear_validation_v2_fields.npz",
        {
            "coordinate_scales": (69,),
            "base_times_s": (513,),
            "near_onset_a1_final_active_delta": (16, 69),
            "near_onset_a2_final_active_delta": (16, 69),
            "terminal_a1_final_active_delta": (16, 69),
            "terminal_a2_final_active_delta": (16, 69),
        },
    )
    require(bool(operator_contract["contract_pass"]), f"operator array contract failed: {operator_contract}")
    require(bool(nonlinear_contract["contract_pass"]), f"nonlinear field contract failed: {nonlinear_contract}")

    environment_dir = RELEASE / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    requirements = environment_dir / "requirements-lock.txt"
    requirements.write_text(public_requirements_lock(), encoding="utf-8")
    copied.append(requirements)
    git_revision = command_output(["git", "rev-parse", "HEAD"])
    git_status = command_output(["git", "status", "--short"])
    environment = {
        "schema": "IJP_FINITE_TIME_ENVIRONMENT_V1",
        "python_executable": Path(sys.executable).name,
        "python_version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "blas_lapack_configuration": sanitize_runtime_configuration(
            np.show_config(mode="dicts")
        ),
        "git_revision": git_revision,
        "git_status_at_packaging": git_status,
        "requirements_lock_sha256": sha256(requirements),
        "deterministic_seeds": {"nested_sobol_search": 20260829},
    }
    environment_json = environment_dir / "environment.json"
    environment_json.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copied.append(environment_json)

    readme = RELEASE / "README.md"
    readme.write_text(
        "# Finite-time HCP perturbation reproducibility release V1\n\n"
        "This archive supports the finite-time HCP crystal-plasticity manuscript.\n\n"
        "## Scope\n\n"
        "- `data/05_results/ijp_strengthening_evidence_v1_arrays.npz` contains every "
        "69 x 69 complex generator at all 129 reference checkpoints for the three "
        "reported baseline branches, their full complex leading input/output singular "
        "vectors, complete observed singular spectra, propagators, three-orientation "
        "generators, nested Sobol samples, anchor-assisted audit data, and positive controls.\n"
        "- `ijp_operator_consistency_v2.*` records the like-for-like full-state audit, "
        "norm--selector matrix, commutators, modal conditioning and eigenvector rotation.\n"
        "- `ijp_singular_vector_nonlinear_validation_v2.*` records nonlinear validation "
        "of the actual near-onset and terminal optimal inputs.\n"
        "- `optimizer_trajectories.csv` contains every retained objective evaluation for "
        "all independent local-search starts.\n"
        "- `ijp_material_parameter_provenance_v1.csv` distinguishes literature constraints, "
        "repository verification values, model controls, and numerical controls.\n"
        "- `environment/` freezes the runtime and Python package set.\n"
        "- `manuscript/` contains the audited PDF, its edited source sections, the "
        "generated parameter table, the strengthening figure, and the six-item "
        "acceptance audit.\n"
        "- `MANIFEST.sha256` authenticates every packaged file.\n\n"
        "## Claim boundary\n\n"
        "The package supports like-for-like full-state finite-time/frozen discrimination, "
        "norm--selector robustness and an anchor-assisted direction--wavenumber audit. It does not claim specimen "
        "calibration, nonlinear material band width, propagation resistance, an analytic "
        "global optimum proof, or a strict HCP-to-Bai analytical degeneration.\n",
        encoding="utf-8",
    )
    copied.append(readme)

    manifest_path = RELEASE / "MANIFEST.sha256"
    release_files = sorted(
        path for path in RELEASE.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest_lines = [
        f"{sha256(path)}  {path.relative_to(RELEASE).as_posix()}" for path in release_files
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    manifest_hash = sha256(manifest_path)

    # shutil.make_archive appends .zip to the base name and replaces only that
    # exact in-scope artifact on rerun.
    created_archive = Path(
        shutil.make_archive(
            str(ARCHIVE.with_suffix("")), "zip", root_dir=RELEASE.parent, base_dir=RELEASE.name
        )
    )
    require(created_archive.resolve() == ARCHIVE.resolve(), "unexpected archive target")
    with zipfile.ZipFile(ARCHIVE, "r") as stream:
        archived_files = [name for name in stream.namelist() if not name.endswith("/")]
        bad_member = stream.testzip()
    expected_archive_count = len(release_files) + 1
    gates = {
        "array_contract_pass": bool(contract["contract_pass"]),
        "operator_array_contract_pass": bool(operator_contract["contract_pass"]),
        "nonlinear_field_contract_pass": bool(nonlinear_contract["contract_pass"]),
        "manifest_covers_every_release_file": len(manifest_lines) == len(release_files),
        "zip_member_count_matches": len(archived_files) == expected_archive_count,
        "zip_crc_test_pass": bad_member is None,
        "environment_lock_present": requirements.is_file() and environment_json.is_file(),
        "optimizer_trace_present": (RELEASE / "data/05_results/ijp_strengthening_evidence_v1/optimizer_trajectories.csv").is_file(),
        "audited_manuscript_present": (
            RELEASE / "manuscript/06_manuscript/ijp_spectral_hcp/main_finite_time.pdf"
        ).is_file(),
    }
    result = {
        "schema": "IJP_FINITE_TIME_REPRODUCIBILITY_RELEASE_V1",
        "status": "PUBLICATION_READY_LOCAL_ARCHIVE" if all(gates.values()) else "RELEASE_VERIFICATION_FAILED",
        "release_directory": RELEASE.relative_to(ROOT).as_posix(),
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "manifest_sha256": manifest_hash,
        "release_file_count_excluding_manifest": len(release_files),
        "array_contract": contract,
        "operator_array_contract": operator_contract,
        "nonlinear_field_contract": nonlinear_contract,
        "gates": gates,
        "publication_note": (
            "The archive is ready for repository/DOI deposition but has not been uploaded by this script."
        ),
    }
    RECEIPT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(RECEIPT), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
