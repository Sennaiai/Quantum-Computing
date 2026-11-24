"""Offline molecular CD/QSVT orchestration and reproducible API input packages.

No operation in this module submits a job. The candidate Beryllium program
requires native gate, compiler, deployment, and service-contract verification.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "runtime"
QBE_SOURCE_DIR = ROOT / "qbe_modules"
for candidate in (ROOT / "src", ROOT / "python_classes"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from admet_curation import CurationConfig, FIXTURE_NAME, molecule_problem_input, run_complete_curation
from artifact_paths import validate_output_dir
from classical_preprocessing import AGPSettings, recalculate_counterdiabatic_hamiltonian
from hamiltonian_validation import HamiltonianValidation, finite_real, positive_integer
from qsvt_helper import QSVTHelper
from quantum_reference import verify_quantum_reference, verify_spectral_reference
from runtime_input_protocol import (atomic_write_json, build_runtime_input_document, canonical_json,
                                    validate_runtime_input_document)
from runtime_sync_validation import QBE_MODULES, verify_runtime_synchronization


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def build_validator(payload):
    validator = HamiltonianValidation(payload)
    validator.validate()
    return validator


def build_molecular_payload(*, data_mode="fixture", dataset=None, sample_index=13,
                            interpolation=0.5, lambda_dot=1.0, config=None, datasets=None):
    """Run the canonical curation and bind one molecule; retain all interactions."""
    if dataset is None:
        dataset = (FIXTURE_NAME if data_mode == "fixture" else
                   next(iter(datasets)) if data_mode == "provided" and datasets and len(datasets) == 1 else
                   "Provided_ADMET" if data_mode == "provided" else "BBB_Martins")
    curation = run_complete_curation(mode=data_mode, config=config, datasets=datasets,
                                     dataset_names=(dataset,), cache_path=ROOT / "artifacts" / "tdc")
    adapter, theta, provenance = molecule_problem_input(curation, dataset, sample_index)
    payload = recalculate_counterdiabatic_hamiltonian(
        adapter, dataset, theta,
        AGPSettings(interpolation=interpolation, lambda_dot=lambda_dot),
    )
    payload.update(sample_index=sample_index, curation_provenance=provenance,
                   curation_config=curation["config"], data_mode=data_mode,
                   fixture_labels_are_synthetic=curation["fixture_labels_are_synthetic"])
    build_validator(payload)
    return payload


def prepare_qsvt_settings(validator, physical_time=0.1, epsilon=1e-6):
    return QSVTHelper(validator, physical_time=physical_time, epsilon=epsilon).to_settings()


def source_hashes():
    paths = []
    for directory, pattern in (("src", "*.py"), ("python_classes", "*.py"),
                               ("qbe_modules", "*.qbe"), ("pipeline", "*.ipynb"),
                               ("tests", "*.py")):
        paths.extend((ROOT / directory).glob(pattern))
    paths.extend(p for p in (ROOT / "ADMET_Triple_Alpha_End_to_End.ipynb",
                             ROOT / "requirements.txt", ROOT / "environment.yml",
                             ROOT / "run_checks.py", ROOT / "run_benchmarks.py") if p.is_file())
    return {str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes()) for p in sorted(paths)}


ENVIRONMENT_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "rdkit", "npeet-plus",
                        "pyqsp", "requests", "nbformat", "nbclient", "ipykernel")


def environment_versions():
    return {"python": platform.python_version(), **{p: version(p) for p in ENVIRONMENT_PACKAGES}}


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_FLAGS = ("compiler_verified", "gate_convention_verified", "deployment_verified",
                 "service_contract_verified", "live_api_execution_allowed")
_MANIFEST_KEYS = {"schema_version", "validation_scope", *_POLICY_FLAGS, "module_hashes",
                  "source_hashes", "environment", "mathematical_reference",
                  "local_contract_status", "file_hashes"}
_BUNDLE_FILES = {"hamiltonian.json", "qsvt_settings.json", "runtime_inputs.json"}
_CORE_SOURCE_FILES = {"src/admet_submission_entry.py", "src/artifact_paths.py", "src/admet_curation.py",
                      "src/classical_preprocessing.py", "src/runtime_input_protocol.py",
                      "src/runtime_sync_validation.py", "src/quantum_reference.py",
                      "python_classes/hamiltonian_validation.py", "python_classes/qsvt_helper.py",
                      "requirements.txt", "environment.yml", "run_checks.py"}


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _hash_mapping(value, name, *, exact_names=None):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty hash mapping")
    if exact_names is not None and set(value) != set(exact_names):
        raise ValueError(f"{name} file list differs from schema")
    for path, digest in value.items():
        if (not isinstance(path, str) or not path or "\\" in path
                or Path(path).is_absolute() or any(part in {"", ".", ".."} for part in path.split("/"))
                or not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None):
            raise ValueError(f"Invalid path or digest in {name}")


def _verify_manifest_schema(manifest):
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Bundle manifest fields differ from schema")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2:
        raise ValueError("Unsupported bundle schema")
    if manifest["validation_scope"] != "offline mathematical and local runtime contract":
        raise ValueError("Unsupported bundle validation scope")
    if any(manifest[key] is not False for key in _POLICY_FLAGS):
        raise ValueError("Offline bundle cannot claim native validation or permit live execution")
    if manifest["local_contract_status"] != "local_contract_verified":
        raise ValueError("Invalid local runtime contract status")
    _hash_mapping(manifest["file_hashes"], "Bundle manifest", exact_names=_BUNDLE_FILES)
    _hash_mapping(manifest["module_hashes"], "module_hashes", exact_names=QBE_MODULES)
    _hash_mapping(manifest["source_hashes"], "source_hashes")
    if not _CORE_SOURCE_FILES <= manifest["source_hashes"].keys():
        raise ValueError("Recorded source hashes omit required pipeline files")
    for name, digest in manifest["module_hashes"].items():
        if manifest["source_hashes"].get(f"qbe_modules/{name}") != digest:
            raise ValueError("Module hashes differ from the recorded source hashes")
    environment = manifest["environment"]
    if (not isinstance(environment, dict) or not {"python", *ENVIRONMENT_PACKAGES} <= environment.keys()
            or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                   for key, value in environment.items())):
        raise ValueError("Invalid recorded environment")


def _verify_reference_metadata(reference, settings):
    if not isinstance(reference, dict) or reference != settings.get("mathematical_reference"):
        raise ValueError("Manifest and settings mathematical references differ")
    state = reference.get("status")
    if not isinstance(state, str):
        raise ValueError("Mathematical reference status must be a string")
    if state == "skipped_explicitly":
        if not isinstance(reference.get("reason"), str) or not reference["reason"]:
            raise ValueError("Skipped mathematical reference requires a reason")
    elif state in {"verified_mathematical_reference", "verified_spectral_reference"}:
        if reference.get("native_compiler_verified") is not False:
            raise ValueError("Mathematical reference cannot certify native compilation")
        error = finite_real(reference.get("recovered_evolution_error"), "reference error")
        bound = finite_real(reference.get("certified_recovered_evolution_error_bound"), "reference error bound")
        if error < 0 or bound != settings["recovered_evolution_error_bound"] or error > bound + 1e-11:
            raise ValueError("Recorded mathematical reference exceeds or differs from its error bound")
    else:
        raise ValueError("Unknown mathematical reference status")


def verify_runtime_bundle(bundle_dir, *, require_current_sources=True, expected_package_id=None):
    """Verify the content address and semantic identity of an offline package.

    Historical mode omits only comparison with current source files. It still
    checks the Hamiltonian, settings, stream, policy, and reference relationships.
    Pending directories require their intended final package ID explicitly.
    """
    if not isinstance(require_current_sources, bool):
        raise ValueError("require_current_sources must be a Boolean")
    bundle_dir = Path(bundle_dir)
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise ValueError("Bundle must be a regular directory")
    if {path.name for path in bundle_dir.iterdir()} != _BUNDLE_FILES | {"manifest.json"}:
        raise ValueError("Bundle entries differ from schema")
    def read(name):
        path = bundle_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Bundle entries must be regular files")
        data = path.read_bytes()
        return _strict_json(data.decode("utf-8")), data
    manifest, _ = read("manifest.json")
    _verify_manifest_schema(manifest)
    package_id = bundle_dir.name if expected_package_id is None else expected_package_id
    if (expected_package_id is not None and bundle_dir.name != expected_package_id
            and not bundle_dir.name.startswith(".pending-")):
        raise ValueError("Only pending directories may use an explicit package identity")
    if (not isinstance(package_id, str) or _DIGEST.fullmatch(package_id) is None
            or sha256_bytes(canonical_json(manifest).encode()) != package_id):
        raise ValueError("Bundle content address differs from its manifest")
    documents = {}
    for name, digest in manifest["file_hashes"].items():
        value, data = read(name)
        if sha256_bytes(data) != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
        documents[name] = value
    payload, settings, document = (documents[name] for name in
        ("hamiltonian.json", "qsvt_settings.json", "runtime_inputs.json"))
    if not isinstance(settings, dict):
        raise ValueError("QSVT settings must be an object")
    validator = build_validator(payload)
    validate_runtime_input_document(document)
    if build_runtime_input_document(payload, settings, validator) != document:
        raise ValueError("Runtime document differs from its Hamiltonian and phases")
    _verify_reference_metadata(manifest["mathematical_reference"], settings)
    native = settings.get("native_validation")
    if not isinstance(native, dict) or native.get("status") != "unverified":
        raise ValueError("Offline settings require explicitly unverified native status")
    if native.get("compiler_verified") is not False or native.get("live_api_execution_allowed") is not False:
        raise ValueError("Offline settings must explicitly deny compilation and live execution")
    if any(key in native and native[key] is not False for key in _POLICY_FLAGS):
        raise ValueError("Offline settings cannot claim native validation")
    if require_current_sources:
        if source_hashes() != manifest["source_hashes"]:
            raise ValueError("Current sources differ from the bundle's recorded sources")
        rebuilt = verify_runtime_synchronization(payload, QBE_SOURCE_DIR, settings, validator,
                                               expected_module_hashes=manifest["module_hashes"])
        if rebuilt["runtime_document"] != document:
            raise ValueError("Runtime document differs from its local source contract")
    return manifest


def prepare_runtime_inputs(payload, output_dir=None, physical_time=0.1, epsilon=1e-6,
                           *, dense_reference=True, max_reference_dimension=256):
    """Write an immutable, content-addressed package; never overwrite QBE source.

    Bounded dense checks are mandatory by default. Larger exploratory runs may
    explicitly skip that exponential reference, with the omission in metadata.
    """
    output_root = validate_output_dir(output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
                                      project_root=ROOT)
    if not isinstance(dense_reference, bool):
        raise ValueError("dense_reference must be a Boolean")
    max_reference_dimension = positive_integer(max_reference_dimension, "max_reference_dimension")
    physical_time = finite_real(physical_time, "physical_time")
    epsilon = finite_real(epsilon, "epsilon")
    validator = build_validator(payload)
    # Reject unsupported metadata/containers before phase synthesis or dense
    # allocation; accepted scientific numeric scalars remain strict JSON.
    canonical_json(payload)
    settings = prepare_qsvt_settings(validator, physical_time, epsilon)
    if dense_reference:
        circuit_dimension = 2**(validator.num_system_qubits+validator.num_index_qubits)
        if circuit_dimension <= max_reference_dimension:
            reference = verify_quantum_reference(payload, settings, max_dimension=max_reference_dimension)
        else:
            reference = verify_spectral_reference(payload, settings)
    else:
        reference = {"status": "skipped_explicitly", "reason": "bounded dense reference disabled by caller"}
    synchronization = verify_runtime_synchronization(payload, QBE_SOURCE_DIR, settings, validator)
    settings["mathematical_reference"] = reference
    document = synchronization["runtime_document"]
    documents = {"hamiltonian.json": payload, "qsvt_settings.json": settings,
                 "runtime_inputs.json": document}
    hashes = source_hashes()
    manifest = {
        "schema_version": 2, "validation_scope": "offline mathematical and local runtime contract",
        "compiler_verified": False, "gate_convention_verified": False,
        "deployment_verified": False, "service_contract_verified": False,
        "live_api_execution_allowed": False, "module_hashes": synchronization["module_hashes"],
        "source_hashes": hashes, "environment": environment_versions(),
        "mathematical_reference": reference, "local_contract_status": synchronization["status"],
        "file_hashes": {name: sha256_bytes((canonical_json(value)+"\n").encode())
                        for name, value in documents.items()},
    }
    package_id = sha256_bytes(canonical_json(manifest).encode())
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / package_id
    if bundle.exists():
        verify_runtime_bundle(bundle)
    else:
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=output_root))
        try:
            for name, value in documents.items():
                (temporary / name).write_text(canonical_json(value)+"\n", encoding="utf-8")
            atomic_write_json(temporary / "manifest.json", manifest)
            verify_runtime_bundle(temporary, expected_package_id=package_id)
            try:
                temporary.rename(bundle)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                # Another writer may have published the identical immutable ID.
                verify_runtime_bundle(bundle)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    settings["synchronization"] = synchronization
    return bundle, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-mode", choices=("fixture", "provided", "tdc"), default="fixture")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-csv", type=Path, help="User-provided Drug,Y CSV; no download")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Explicit reduced feature profile; original nonfixture default is 100")
    parser.add_argument("--sample-index", type=int, default=13)
    parser.add_argument("--payload", type=Path, help="Validated JSON H_CD instead of curation (never pickle)")
    parser.add_argument("--interpolation", type=float, default=0.5)
    parser.add_argument("--lambda-dot", type=float, default=1.0)
    parser.add_argument("--physical-time", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-dense-reference", action="store_true",
                        help="Explicitly skip the exponential reference for large exploratory instances")
    args = parser.parse_args(argv)
    if args.payload and args.data_csv:
        parser.error("--payload and --data-csv are mutually exclusive")
    if args.data_mode == "provided" and args.data_csv is None and not args.payload:
        parser.error("--data-mode provided requires --data-csv")
    if args.payload and args.top_k is not None:
        parser.error("--top-k applies to curation, not an imported Hamiltonian")
    if args.payload:
        payload = _strict_json(args.payload.read_text())
    else:
        tables, mode, dataset = None, args.data_mode, args.dataset
        if args.data_csv is not None:
            import pandas as pd
            mode, dataset = "provided", dataset or "Provided_ADMET"
            tables = {dataset: pd.read_csv(args.data_csv)}
        config = CurationConfig(top_k=args.top_k) if args.top_k is not None else None
        payload = build_molecular_payload(data_mode=mode, dataset=dataset, datasets=tables, config=config,
            sample_index=args.sample_index, interpolation=args.interpolation, lambda_dot=args.lambda_dot)
    bundle, settings = prepare_runtime_inputs(payload, args.output_dir, args.physical_time,
        args.epsilon, dense_reference=not args.skip_dense_reference)
    print(json.dumps({"bundle": str(bundle), "status": settings["synchronization"]["status"],
        "terms": len(payload["pauli_terms"]), "lambda": settings["lambda_value"],
        "recovered_error_bound": settings["recovered_evolution_error_bound"],
        "success_probability_lower_bound": settings["success_probability_lower_bound"],
        "live_api_execution_allowed": False}, indent=2))


if __name__ == "__main__":
    main()
