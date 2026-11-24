"""Prepare every molecule against one frozen training-only encoding.

Identical feature vectors may share immutable physics bundles. Molecule/task/job
identities never share their receipts. Deployment families group source-level
dimensions only; actual native capacities and input/output bindings require
independent compiler validation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import re
from pathlib import Path
import tempfile
import shutil

from admet_submission_entry import prepare_runtime_inputs, source_hashes, verify_runtime_bundle
from artifact_paths import validate_output_dir
from classical_preprocessing import AGPSettings, recalculate_counterdiabatic_hamiltonian
from hamiltonian_validation import finite_real, positive_integer
from result_contract import build_feature_schema, validate_feature_schema, validate_task
from runtime_input_protocol import atomic_write_json, canonical_json, decode_runtime_input_values


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _execution_profile(encoding_fit_id, settings, sources):
    """Bind learned measurement features to their physical experiment."""
    return {"schema_version": 1, "encoding_fit_id": encoding_fit_id,
        "interpolation": settings["interpolation"], "lambda_dot": settings["lambda_dot"],
        "physical_time": settings["physical_time"], "epsilon": settings["epsilon"],
        "initialization": "RY(pi*theta), wire0 first", "runtime_protocol_version": 2,
        "source_hashes": sources}


@dataclass(frozen=True)
class BatchSettings:
    interpolation: float = .5
    lambda_dot: float = 1.
    physical_time: float = .1
    epsilon: float = 1e-6
    shots_requested: int = 10000
    max_molecules: int = 25000
    max_unique_physics: int = 5000
    max_reference_dimension: int = 256

    def validate(self):
        AGPSettings(interpolation=self.interpolation, lambda_dot=self.lambda_dot).validate()
        time, epsilon = finite_real(self.physical_time), finite_real(self.epsilon)
        if time <= 0 or not 1e-10 <= epsilon < 1:
            raise ValueError("Batch time must be positive and epsilon within the supported float64 range")
        for name in ("shots_requested", "max_molecules", "max_unique_physics", "max_reference_dimension"):
            positive_integer(getattr(self,name),name)
        if self.shots_requested>10_000_000 or self.max_molecules>100_000 or self.max_unique_physics>100_000:
            raise ValueError("Batch settings exceed the supported bounded budgets")


def _physics_bundle(encoding, theta, settings, output, cache_key):
    endpoint=encoding["endpoint"]
    payload=recalculate_counterdiabatic_hamiltonian(encoding["adapter"],endpoint,theta,
        AGPSettings(interpolation=settings.interpolation,lambda_dot=settings.lambda_dot))
    payload.update(encoding_fit_id=encoding["encoding_fit_id"],physics_cache_key=cache_key,
        data_mode="frozen_training_encoding",fixture_labels_are_synthetic=False,
        curation_provenance={"fit_scope":"training_only","encoding_fit_id":encoding["encoding_fit_id"],
            "train_molecule_ids_sha256":encoding["train_molecule_ids_sha256"],
            "local_field_binding":"this transformed molecule's theta; no labels read"})
    directory,qsvt=prepare_runtime_inputs(payload,output,physical_time=settings.physical_time,
        epsilon=settings.epsilon,max_reference_dimension=settings.max_reference_dimension)
    manifest=verify_runtime_bundle(directory)
    document=qsvt["synchronization"]["runtime_document"]
    decoded=decode_runtime_input_values(document["inputs"])
    phase_settings_sha=manifest["file_hashes"]["qsvt_settings.json"]
    shape={"runtime_protocol_version":document["protocol_version"],
        "num_system_qubits":decoded["num_system_qubits"],"num_index_qubits":decoded["num_index_qubits"],
        "num_terms":decoded["num_terms"],"num_cosine_phases":decoded["num_cosine_phases"],
        "num_sine_phases":decoded["num_sine_phases"],"module_hashes":manifest["module_hashes"],
        "initialization":"RY(pi*theta), wire0 first", "result_layout":"branch,real,lambda,index_MSB,system_wire0_first"}
    family={**shape,"deployment_family_id":_hash(shape),
        "native_capacity_verified":False,"native_input_binding_verified":False,"native_output_mapping_verified":False}
    return {"bundle_id":directory.name,"inputs_sha256":document["inputs_sha256"],
        "phase_settings_sha256":phase_settings_sha,"deployment_family":family,
        "lambda_value":qsvt["lambda_value"],"num_system_qubits":decoded["num_system_qubits"],
        "num_index_qubits":decoded["num_index_qubits"],"input_count":document["input_count"],
        "reference":qsvt["mathematical_reference"],"recovered_evolution_error_bound":qsvt["recovered_evolution_error_bound"],
        "success_probability_lower_bound":qsvt["success_probability_lower_bound"]}


def prepare_batch(encoding_schema, transformed, output_dir, settings=None):
    """Return ``(batch_directory, manifest)`` without any API operation.

    Every transformed canonical identity is represented exactly once. No prefix
    sampling, term truncation, or unknown-molecule omission is permitted.
    Physics is recomputed for each distinct theta under fixed train-only H_A.
    """
    from supervised_encoding import validate_frozen_schema
    validate_frozen_schema(encoding_schema)
    settings=BatchSettings() if settings is None else settings
    if not isinstance(settings,BatchSettings):raise ValueError("settings must be BatchSettings")
    settings.validate()
    output=validate_output_dir(output_dir)
    if (not isinstance(transformed,dict) or transformed.get("schema_id")!=encoding_schema["encoding_fit_id"]
            or transformed.get("endpoint")!=encoding_schema["endpoint"] or not isinstance(transformed.get("records"),list)):
        raise ValueError("Transformed molecules belong to a different frozen encoding")
    records=transformed["records"]
    if not records or len(records)>settings.max_molecules:
        raise ValueError("Molecule batch is empty or exceeds its explicit budget")
    ids=[record.get("molecule_id") for record in records]
    if any(not isinstance(value,str) or re.fullmatch(r"[0-9a-f]{64}",value) is None for value in ids) or len(set(ids))!=len(ids):
        raise ValueError("Canonical molecule identities must be unique SHA identifiers")
    selected=encoding_schema["active_original_features"]
    before=source_hashes()
    profile=_execution_profile(encoding_schema["encoding_fit_id"],asdict(settings),before)
    feature_schema=build_feature_schema(encoding_schema["endpoint"],encoding_schema["encoding_fit_id"],selected,
        encoding_schema["readout_supports"],execution_profile_id=_hash(profile))
    cache, tasks, physics_keys={},[],[]
    # Preflight all theta/group counts before synthesizing or publishing bundles.
    for record in records:
        if record.get("endpoint")!=encoding_schema["endpoint"] or record.get("encoding_fit_id")!=encoding_schema["encoding_fit_id"]:
            raise ValueError("A transformed molecule belongs to a different endpoint/encoding")
        theta=[finite_real(x,"molecular theta") for x in record.get("theta",[])]
        if len(theta)!=len(selected) or any(x<0 or x>1 for x in theta):
            raise ValueError("Molecular theta must match the frozen [0,1] feature map")
        physics_keys.append(_hash({"encoding_fit_id":encoding_schema["encoding_fit_id"],"theta":theta,"settings":asdict(settings)}))
    if len(set(physics_keys))>settings.max_unique_physics:
        raise ValueError("Unique physics budget exceeded; no molecule was silently omitted")
    for record,key in zip(records,physics_keys):
        if key not in cache:
            cache[key]=_physics_bundle(encoding_schema,record["theta"],settings,output/"bundles",key)
        physics=cache[key]
        task={"schema_version":1,"molecule_id":record["molecule_id"],"endpoint":encoding_schema["endpoint"],
            "encoding_fit_id":encoding_schema["encoding_fit_id"],"feature_schema_id":feature_schema["feature_schema_id"],
            **{name:physics[name] for name in ("bundle_id","inputs_sha256","phase_settings_sha256","lambda_value","num_system_qubits","num_index_qubits")},
            "deployment_family_id":physics["deployment_family"]["deployment_family_id"],
            "shots_requested":settings.shots_requested,
            "source_row_positions":list(record.get("source_row_positions",[])),
            "source_record_ids":list(record.get("source_record_ids",[]))}
        task["task_id"]=_hash(task);validate_task(task);tasks.append(task)
    if source_hashes()!=before:
        raise RuntimeError("Sources changed during all-molecule preparation; rerun on a stable tree")
    groups=Counter(physics_keys)
    families={p["deployment_family"]["deployment_family_id"]:p["deployment_family"] for p in cache.values()}
    manifest={"schema_version":1,"endpoint":encoding_schema["endpoint"],"encoding_fit_id":encoding_schema["encoding_fit_id"],
        "feature_schema":feature_schema,"execution_profile":profile,"settings":asdict(settings),"source_hashes":before,
        "source_tree_sha256":_hash(before),"molecule_count":len(tasks),"unique_physics_count":len(cache),
        "physics_reuse_count":len(tasks)-len(cache),"tasks":sorted(tasks,key=lambda t:t["molecule_id"]),
        "physics_groups":[{"physics_key":key,"molecule_count":groups[key],**cache[key]} for key in sorted(cache)],
        "deployment_families":[families[key] for key in sorted(families)],
        "live_api_calls":0,"native_compiler_verified":False,"native_input_binding_verified":False,
        "native_output_mapping_verified":False,"scope":"all supplied canonical identities, with shared physics only for identical fitted theta/settings; no native execution claim"}
    manifest["batch_id"]=_hash(manifest)
    validate_batch_manifest(manifest)
    directory=output/manifest["batch_id"]
    if directory.exists():
        import json
        existing=json.loads((directory/"batch_manifest.json").read_text())
        if existing!=manifest:raise ValueError("Existing batch identity differs from its contents")
    else:
        output.mkdir(parents=True,exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix=".pending-batch-",dir=output))
        try:
            atomic_write_json(temporary/"encoding_schema.json",encoding_schema)
            atomic_write_json(temporary/"feature_schema.json",feature_schema)
            atomic_write_json(temporary/"batch_manifest.json",manifest)
            temporary.rename(directory)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    return directory,manifest


def validate_batch_manifest(manifest, *, output_root=None):
    fields={"schema_version","endpoint","encoding_fit_id","feature_schema","execution_profile","settings","source_hashes",
        "source_tree_sha256","molecule_count","unique_physics_count","physics_reuse_count","tasks",
        "physics_groups","deployment_families","live_api_calls","native_compiler_verified",
        "native_input_binding_verified","native_output_mapping_verified","scope","batch_id"}
    if not isinstance(manifest,dict) or set(manifest)!=fields or type(manifest.get("schema_version")) is not int or manifest["schema_version"]!=1:
        raise ValueError("Unsupported batch manifest")
    BatchSettings(**manifest["settings"]).validate()
    if not isinstance(manifest["source_hashes"],dict) or _hash(manifest["source_hashes"])!=manifest["source_tree_sha256"]:
        raise ValueError("Batch source identity is inconsistent")
    validate_feature_schema(manifest["feature_schema"])
    profile=_execution_profile(manifest["encoding_fit_id"],manifest["settings"],manifest["source_hashes"])
    if manifest["execution_profile"]!=profile or manifest["feature_schema"]["execution_profile_id"]!=_hash(profile) or manifest["feature_schema"]["execution_profile_kind"]!="declared_physics":
        raise ValueError("Batch and physical measurement profile differ")
    if _hash({k:v for k,v in manifest.items() if k!="batch_id"})!=manifest.get("batch_id"):
        raise ValueError("Batch identity differs from its contents")
    for key in ("native_compiler_verified","native_input_binding_verified","native_output_mapping_verified"):
        if manifest.get(key) is not False:raise ValueError("Offline batch cannot attest native compatibility")
    if manifest.get("live_api_calls")!=0:raise ValueError("Offline batch cannot contain live executions")
    if manifest["endpoint"]!=manifest["feature_schema"]["endpoint"] or manifest["encoding_fit_id"]!=manifest["feature_schema"]["encoding_fit_id"]:
        raise ValueError("Batch and readout encoding differ")
    families={}
    for family in manifest["deployment_families"]:
        flags={"native_capacity_verified","native_input_binding_verified","native_output_mapping_verified"}
        if any(family.get(k) is not False for k in flags):
            raise ValueError("Source family cannot attest native capacity or bindings")
        shape={k:v for k,v in family.items() if k not in flags|{"deployment_family_id"}}
        if _hash(shape)!=family.get("deployment_family_id") or family["deployment_family_id"] in families:
            raise ValueError("Deployment family identity is inconsistent")
        families[family["deployment_family_id"]]=family
    groups={}
    for physics in manifest["physics_groups"]:
        if physics["bundle_id"] in groups or physics["deployment_family"]["deployment_family_id"] not in families:
            raise ValueError("Repeated or unknown physics/deployment group")
        if physics["deployment_family"]!=families[physics["deployment_family"]["deployment_family_id"]]:
            raise ValueError("Physics group differs from its deployment family")
        measured=finite_real(physics["reference"]["recovered_evolution_error"])
        bound=finite_real(physics["recovered_evolution_error_bound"])
        if not 0<=measured<=bound+1e-11 or not 0<=bound<=manifest["settings"]["epsilon"]:
            raise ValueError("Physics group exceeds its evolution error budget")
        positive_integer(physics["molecule_count"],"physics molecule count")
        groups[physics["bundle_id"]]=physics
    tasks=manifest.get("tasks")
    if not isinstance(tasks,list) or len(tasks)!=manifest["molecule_count"] or not tasks:
        raise ValueError("Batch molecule/task count mismatch")
    ids=set();task_ids=set()
    for task in tasks:
        validate_task(task)
        if task["molecule_id"] in ids or task["task_id"] in task_ids:raise ValueError("Repeated molecule/task identity")
        ids.add(task["molecule_id"]);task_ids.add(task["task_id"])
        if task["endpoint"]!=manifest["endpoint"] or task["encoding_fit_id"]!=manifest["encoding_fit_id"] or task["feature_schema_id"]!=manifest["feature_schema"]["feature_schema_id"]:
            raise ValueError("Task belongs to another frozen endpoint/readout schema")
        if task["bundle_id"] not in groups or task["deployment_family_id"] not in families:
            raise ValueError("Task refers to an unknown physics bundle or deployment family")
        physics=groups[task["bundle_id"]]
        for key in ("inputs_sha256","phase_settings_sha256","lambda_value","num_system_qubits","num_index_qubits"):
            if task[key]!=physics[key]:raise ValueError("Task and physics metadata differ")
        if task["deployment_family_id"]!=physics["deployment_family"]["deployment_family_id"] or task["shots_requested"]!=manifest["settings"]["shots_requested"]:
            raise ValueError("Task family or shot budget differs from batch settings")
    frequencies=Counter(task["bundle_id"] for task in tasks)
    if set(frequencies)!=set(groups) or any(frequencies[key]!=group["molecule_count"] for key,group in groups.items()):
        raise ValueError("Physics group molecule coverage differs from tasks")
    if manifest["unique_physics_count"]!=len(groups) or manifest["physics_reuse_count"]!=len(tasks)-len(groups):
        raise ValueError("Physics reuse accounting differs from complete molecule coverage")
    if output_root is not None:
        root=Path(output_root)
        for bundle_id,physics in groups.items():
            directory=root/"bundles"/bundle_id
            physical_manifest=verify_runtime_bundle(directory)
            import json
            document=json.loads((directory/"runtime_inputs.json").read_text())
            decoded=decode_runtime_input_values(document["inputs"])
            if document["inputs_sha256"]!=physics["inputs_sha256"] or physical_manifest["file_hashes"]["qsvt_settings.json"]!=physics["phase_settings_sha256"] or not math.isclose(decoded["lambda_value"],physics["lambda_value"],rel_tol=1e-12):
                raise ValueError("Task differs from its immutable physics bundle")
            if physical_manifest["source_hashes"]!=manifest["source_hashes"] or physical_manifest["mathematical_reference"]!=physics["reference"]:
                raise ValueError("Physics source/reference differs from batch metadata")
            family=physics["deployment_family"]
            for key in ("num_terms","num_system_qubits","num_index_qubits","num_cosine_phases","num_sine_phases"):
                if decoded[key]!=family[key]:raise ValueError("Deployment dimensions differ from the runtime stream")
            if physical_manifest["module_hashes"]!=family["module_hashes"]:
                raise ValueError("Deployment module sources differ from physics bundle")
    if len({t["bundle_id"] for t in tasks})!=manifest["unique_physics_count"]:
        raise ValueError("Unique physics/task count mismatch")
    return manifest
