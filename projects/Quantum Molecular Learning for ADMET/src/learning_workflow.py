"""Prepare all molecules and fit/evaluate a post-API model without leakage.

Preparation performs no quantum API operation. The quantum readout stays
unfitted until validated measurements are supplied. Classical baselines use
the same cleaned identities and strict scaffold partitions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_curation import CurationConfig
from admet_submission_entry import ROOT, QBE_SOURCE_DIR, source_hashes, _strict_json
from artifact_paths import validate_output_dir
from batch_execution import BatchSettings, prepare_batch, validate_batch_manifest
from data_quality import clean_endpoint_table, split_cleaned_dataset, validate_split_manifest
from native_binding import write_native_family, named_inputs
from triple_alpha_client import validate_named_inputs
from post_api_ml import (ReadoutConfig, prepare_readout_spec, fit_classical_baseline,
    evaluate_classical_baseline, fit_correlation_model, evaluate_correlation_model,
    predict_correlation_model, write_model)
from result_contract import validate_job_bindings, decode_result
from runtime_input_protocol import atomic_write_json, canonical_json
from supervised_encoding import fit_hamiltonian_schema, transform_molecules, validate_frozen_schema


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _partition(records, ids):
    wanted = set(ids)
    result = [row for row in records if row["molecule_id"] in wanted]
    if {row["molecule_id"] for row in result} != wanted:
        raise ValueError("Molecular/result records do not completely cover the requested partition")
    return result


def prepare_learning_workflow(table, endpoint, output_dir, *, curation_config=None,
                              batch_settings=None, readout_config=None, split_seed=42):
    """Clean, split, fit on train, and prepare EVERY valid canonical structure."""
    output = validate_output_dir(output_dir)
    before = source_hashes()
    clean = clean_endpoint_table(table, endpoint)
    partitions = split_cleaned_dataset(clean["cleaned"], endpoint=endpoint, seed=split_seed)
    split = partitions["manifest"]
    schema = fit_hamiltonian_schema(partitions["train"], endpoint,
        config=curation_config or CurationConfig(top_k=4))
    transformed = transform_molecules(schema, clean["inference"])
    batch_root = output/"batch"
    batch_directory, batch = prepare_batch(schema, transformed, batch_root, settings=batch_settings)
    if batch["molecule_count"] != clean["audit"]["valid_inference_identities"]:
        raise ValueError("All valid molecular identities must be represented in the API batch")
    all_rows = sorted(position for task in batch["tasks"] for position in task["source_row_positions"])
    expected_rows = sorted(position for positions in clean["inference"]["source_row_positions"] for position in positions)
    if all_rows != expected_rows:
        raise ValueError("Original source-row coverage differs from the prepared molecule batch")
    labels = dict(zip(clean["cleaned"]["molecule_id"], map(int, clean["cleaned"]["Y"])))
    config = ReadoutConfig() if readout_config is None else readout_config
    spec = prepare_readout_spec(schema, batch["feature_schema"], split, config=config)
    baseline_metrics, baseline_models = {}, {}
    for representation in ("theta", "full_counts"):
        model = fit_classical_baseline(
            _partition(transformed["records"], split["train"]), {i:labels[i] for i in split["train"]},
            validation_records=_partition(transformed["records"], split["validation"]),
            validation_labels={i:labels[i] for i in split["validation"]},
            encoding_schema=schema, split_manifest=split, representation=representation, config=config)
        # Test labels enter only this separate evaluation after fitting/selection.
        evaluation = evaluate_classical_baseline(model,
            _partition(transformed["records"], split["test"]), {i:labels[i] for i in split["test"]}, encoding_schema=schema)
        baseline_models[representation] = model
        baseline_metrics[representation] = evaluation
    bindings = []
    for family in batch["deployment_families"]:
        _, binding = write_native_family(family, QBE_SOURCE_DIR, output/"native_families")
        bindings.append(binding)
    by_family={binding["deployment_family_id"]:binding for binding in bindings}
    for physics in batch["physics_groups"]:
        document=_strict_json((batch_root/"bundles"/physics["bundle_id"]/"runtime_inputs.json").read_text())
        pairs=named_inputs(document,by_family[physics["deployment_family"]["deployment_family_id"]])
        validate_named_inputs(pairs)
        if [float(value) for _,value in pairs]!=document["inputs"][5:]:
            raise ValueError("Named API inputs differ from the verified scientific stream")
    if source_hashes() != before:
        raise RuntimeError("Sources changed during supervised preparation; rerun on a stable tree")
    summary = {"schema_version": 1, "status": "awaiting_api_results", "endpoint": endpoint,
        "source_hashes": before, "source_tree_sha256": _digest(before),
        "raw_rows": len(table),
        "cleaned_labelled_identities": len(clean["cleaned"]),
        "all_valid_inference_identities": len(clean["inference"]),
        "quarantined_label_identities": clean["audit"]["conflicting_label_identities"],
        "quarantined_labelled_rows": clean["audit"]["conflicting_labelled_rows_quarantined"],
        "concordant_duplicate_rows_collapsed": clean["audit"]["concordant_duplicate_rows_collapsed"],
        "invalid_smiles_rows": clean["audit"]["invalid_smiles_rows"],
        "invalid_label_rows": clean["audit"]["invalid_label_rows"],
        "split_id": split["split_id"], "split_sizes": {name:len(split[name]) for name in ("train", "validation", "test")},
        "split_class_counts": split["class_counts"], "scaffold_groups_disjoint": True,
        "encoding_fit_id": schema["encoding_fit_id"], "fit_scope": "training_only",
        "selected_original_features": schema["selected_original_features"],
        "transform_diagnostics": transformed["diagnostics"], "batch_id": batch["batch_id"],
        "prepared_molecules": batch["molecule_count"], "unique_physics": batch["unique_physics_count"],
        "physics_reuse_count": batch["physics_reuse_count"], "native_family_count": len(bindings),
        "source_named_input_roundtrips":len(batch["physics_groups"]),
        "maximum_recovered_operator_error": max(p["reference"]["recovered_evolution_error"] for p in batch["physics_groups"]),
        "maximum_recovered_error_bound": max(p["recovered_evolution_error_bound"] for p in batch["physics_groups"]),
        "readout_spec": spec, "classical_test_results": baseline_metrics,
        "quantum_prediction_metrics": None, "quantum_model_fitted": False,
        "native_compiler_verified": False, "native_input_binding_verified": False,
        "native_output_mapping_verified": False, "triple_alpha_calls": 0,
        "interpretation": "Real classical baselines only; quantum model is present but awaits validated API measurements. No synthetic quantum metric is reported as hardware performance."}
    documents = {"summary.json": summary, "data_audit.json": clean["audit"],
        "split_manifest.json": split, "encoding_schema.json": schema,
        "transformed_records.json": transformed, "batch_manifest.json": batch,
        "feature_schema.json": batch["feature_schema"], "readout_spec.json": spec,
        "labels.json": {"endpoint":endpoint, "split_id":split["split_id"], "labels":labels},
        "native_bindings.json": bindings,
        **{f"baseline_{name}.json": value for name,value in baseline_models.items()}}
    files = {name:hashlib.sha256((canonical_json(value)+"\n").encode()).hexdigest() for name,value in documents.items()}
    manifest = {"schema_version":1, "endpoint":endpoint, "batch_id":batch["batch_id"],
        "source_tree_sha256":summary["source_tree_sha256"], "files":files,
        "batch_relative_directory":str(batch_directory.relative_to(output)), "live_api_calls":0}
    manifest["preparation_id"] = _digest(manifest)
    directory = output/manifest["preparation_id"]
    directory.mkdir(parents=True, exist_ok=True)
    for name,value in documents.items():
        atomic_write_json(directory/name, value)
    atomic_write_json(directory/"preparation_manifest.json", manifest)
    load_preparation(directory)
    return directory, summary


def load_preparation(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Supervised preparation must be a regular directory")
    manifest = _strict_json((directory/"preparation_manifest.json").read_text())
    fields={"schema_version","endpoint","batch_id","source_tree_sha256","files","batch_relative_directory","live_api_calls","preparation_id"}
    if not isinstance(manifest,dict) or set(manifest)!=fields or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("preparation_id") != directory.name:
        raise ValueError("Unsupported or mismatched preparation identity")
    if _digest({k:v for k,v in manifest.items() if k != "preparation_id"}) != manifest["preparation_id"]:
        raise ValueError("Preparation content address differs from manifest")
    if manifest.get("live_api_calls") != 0:
        raise ValueError("Preparation cannot contain live executions")
    documents = {}
    expected_files={"summary.json","data_audit.json","split_manifest.json","encoding_schema.json",
        "transformed_records.json","batch_manifest.json","feature_schema.json","readout_spec.json",
        "labels.json","native_bindings.json","baseline_theta.json","baseline_full_counts.json"}
    if not isinstance(manifest["files"],dict) or set(manifest["files"])!=expected_files:
        raise ValueError("Preparation file coverage differs from its schema")
    for name,digest in manifest["files"].items():
        if Path(name).name != name or (directory/name).is_symlink():
            raise ValueError("Invalid preparation filename")
        raw = (directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Supervised preparation file checksum mismatch")
        documents[name] = _strict_json(raw.decode())
    if documents["summary.json"]["source_hashes"] != source_hashes():
        raise ValueError("Preparation sources differ from this implementation; regenerate the fit/batch")
    schema,split=documents["encoding_schema.json"],documents["split_manifest.json"]
    validate_frozen_schema(schema);validate_split_manifest(split)
    labels_doc=documents["labels.json"]
    if (not isinstance(labels_doc,dict) or set(labels_doc)!={"endpoint","split_id","labels"}
            or labels_doc["endpoint"]!=schema["endpoint"] or labels_doc["split_id"]!=split["split_id"]
            or not isinstance(labels_doc["labels"],dict)):
        raise ValueError("Preparation labels differ from the endpoint or split")
    labels=labels_doc["labels"]
    if set(labels)!=set(split["train"]+split["validation"]+split["test"]) or any(type(v) is not int or v not in (0,1) for v in labels.values()):
        raise ValueError("Preparation labels must exactly cover the binary labelled partitions")
    for name in ("train","validation","test"):
        counts={str(label):sum(labels[i]==label for i in split[name]) for label in (0,1)}
        if counts!=split["class_counts"][name]:
            raise ValueError("Preparation label counts differ from the frozen split")
    if _digest([[i,labels[i]] for i in sorted(split["train"])])!=schema["training_labels_sha256"]:
        raise ValueError("Preparation training labels differ from the fitted Hamiltonian")
    batch=documents["batch_manifest.json"]
    if (manifest["endpoint"]!=schema["endpoint"] or manifest["batch_id"]!=batch["batch_id"]
            or manifest["source_tree_sha256"]!=batch["source_tree_sha256"]
            or manifest["batch_relative_directory"]!=f"batch/{batch['batch_id']}"
            or documents["feature_schema.json"]!=batch["feature_schema"]
            or batch["encoding_fit_id"]!=schema["encoding_fit_id"]):
        raise ValueError("Preparation encoding, batch, or physical feature schema differs")
    validate_batch_manifest(documents["batch_manifest.json"], output_root=directory.parent/"batch")
    return documents


def train_post_api(directory, result_document, output_dir, *, config=None):
    """Join verified jobs to molecules, fit train/validation, then evaluate test."""
    documents = load_preparation(directory)
    batch = documents["batch_manifest.json"]
    if not isinstance(result_document, dict) or set(result_document) != {"schema_version", "batch_id", "bindings", "results"}:
        raise ValueError("Expected a versioned batch result document with explicit job bindings")
    if type(result_document["schema_version"]) is not int or result_document["schema_version"] != 1 or result_document["batch_id"] != batch["batch_id"]:
        raise ValueError("Results belong to another prepared batch")
    bindings = validate_job_bindings(result_document["bindings"])
    # Validators may return their input or True; use the checked original list.
    bindings = result_document["bindings"]
    tasks = {task["task_id"]:task for task in batch["tasks"]}
    if {b["task"]["task_id"] for b in bindings} != set(tasks):
        raise ValueError("Job bindings must cover every prepared molecule exactly once")
    by_job = {binding["job_id"]:binding for binding in bindings}
    results = result_document["results"]
    if not isinstance(results, list) or len(results)!=len(by_job) or any(not isinstance(row,dict) for row in results) or {row.get("job_id") for row in results} != set(by_job):
        raise ValueError("Results must cover every bound job without duplicates or omissions")
    readout_config = ReadoutConfig() if config is None else config
    records = []
    for envelope in results:
        binding = by_job[envelope["job_id"]]
        if binding["task"] != tasks[binding["task"]["task_id"]]:
            raise ValueError("Bound job differs from the immutable molecule task")
        records.append(decode_result(binding, envelope, batch["feature_schema"],
            min_heralded_shots=readout_config.minimum_heralded_shots, require_complete=True))
    split, labels = documents["split_manifest.json"], documents["labels.json"]["labels"]
    model = fit_correlation_model(_partition(records, split["train"]), {i:labels[i] for i in split["train"]},
        validation_records=_partition(records, split["validation"]), validation_labels={i:labels[i] for i in split["validation"]},
        encoding_schema=documents["encoding_schema.json"], feature_schema=batch["feature_schema"], split_manifest=split, config=readout_config)
    evaluation = evaluate_correlation_model(model, _partition(records, split["test"]),
        {i:labels[i] for i in split["test"]}, feature_schema=batch["feature_schema"])
    predictions = predict_correlation_model(model, records, feature_schema=batch["feature_schema"])
    output = validate_output_dir(output_dir)
    write_model(output/"correlation_model.json", model)
    report = {"schema_version":1, "status":"fitted", "batch_id":batch["batch_id"],
        "model_id":model["model_id"], "measurement_source":model["data_source"],
        "test_evaluation":evaluation, "predictions":predictions,
        "prediction_count":len(predictions), "triple_alpha_calls":0,
        "scope":"Measurement source is preserved; synthetic records demonstrate software, not hardware ADMET performance"}
    atomic_write_json(output/"post_api_report.json", report)
    return model, report
