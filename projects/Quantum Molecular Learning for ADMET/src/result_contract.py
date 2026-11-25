"""Bound request/result identities and extract heralded Z-correlation features.

The SDK specifies the outer job response, not compiled variable/measurement
names. Native results therefore need an explicit deployment-specific mapping.
No vendor JSON shape, bit order, or repeated-output naming is guessed here.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import math
import re

from hamiltonian_validation import finite_real, positive_integer
from runtime_input_protocol import canonical_json

MAX_SHOTS = 10_000_000
MAX_RESULT_ROWS = 1_000_000
MAX_FEATURES = 4096
SOURCES = {"hardware", "simulator", "synthetic_test"}
UNCERTAINTY_SCOPE = "per raw +/-1 moment conditional on independent heralded shots; not a simultaneous interval or connected-cumulant error bar"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
TASK_FIELDS = {"schema_version", "task_id", "molecule_id", "endpoint", "encoding_fit_id",
    "feature_schema_id", "bundle_id", "inputs_sha256", "phase_settings_sha256",
    "deployment_family_id", "lambda_value", "num_system_qubits", "num_index_qubits",
    "shots_requested", "source_row_positions", "source_record_ids"}


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _digest(value, name):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 identifier")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a nonempty canonical identifier")
    return value


def _feature_name(support, selected):
    return "Z"*len(support)+canonical_json([selected[q] for q in support])


def build_feature_schema(endpoint, encoding_fit_id, selected_features, readout_supports, *, execution_profile_id=None):
    """Declare raw Z moments; connected moments are diagnostics, not ML inputs."""
    selected = list(selected_features)
    if not selected or any(isinstance(x, bool) or not isinstance(x, (str, int)) for x in selected):
        raise ValueError("Selected feature identifiers must be nonempty integers/strings")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected feature identifiers must be unique")
    supports = []
    for key, order in (("singles", 1), ("pairs", 2), ("triads", 3)):
        raw = readout_supports.get(key, [])
        canonical = []
        for item in raw:
            item = [positive_integer(q, "readout wire", allow_zero=True) for q in item]
            if len(item) != order or item != sorted(set(item)) or any(q >= len(selected) for q in item):
                raise ValueError("Readout supports must contain ordered distinct in-range wires")
            canonical.append(tuple(item))
        if len(set(canonical)) != len(canonical):
            raise ValueError("Repeated readout support")
        if key == "singles" and canonical != [(q,) for q in range(len(selected))]:
            raise ValueError("Readout singles must enumerate every system wire in order")
        supports.extend(sorted(canonical))
    if not supports or len(supports) > MAX_FEATURES:
        raise ValueError("Readout feature count exceeds the bounded feature budget")
    schema = {"schema_version": 1, "endpoint": _text(endpoint, "endpoint"),
        "encoding_fit_id": _digest(encoding_fit_id, "encoding_fit_id"),
        "execution_profile_id": (_digest(execution_profile_id,"execution_profile_id") if execution_profile_id is not None
            else _hash({"encoding_fit_id":encoding_fit_id,"profile":"synthetic_unspecified"})),
        "execution_profile_kind": "declared_physics" if execution_profile_id is not None else "synthetic_unspecified",
        "selected_features": selected, "readout_supports": [list(s) for s in supports],
        "feature_names": [_feature_name(s, selected) for s in supports],
        "feature_kind": "raw_z_moments", "basis": "computational_Z_wire0_first",
        "estimator": "empirical raw Z moments conditioned on zero branch, extraction, and index ancillas"}
    schema["feature_schema_id"] = _hash(schema)
    return schema


def validate_feature_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("Feature schema must be an object")
    fields = {"schema_version", "endpoint", "encoding_fit_id", "selected_features", "readout_supports",
              "feature_names", "feature_kind", "basis", "estimator", "feature_schema_id",
              "execution_profile_id","execution_profile_kind"}
    if set(schema) != fields or type(schema["schema_version"]) is not int or schema["schema_version"] != 1:
        raise ValueError("Unsupported correlation feature schema")
    supports = {"singles": [], "pairs": [], "triads": []}
    for row in schema["readout_supports"]:
        if not isinstance(row, list) or len(row) not in (1, 2, 3):
            raise ValueError("Invalid readout support")
        supports[{1:"singles", 2:"pairs", 3:"triads"}[len(row)]].append(row)
    if schema["execution_profile_kind"] not in {"declared_physics","synthetic_unspecified"}:
        raise ValueError("Unknown execution-profile declaration")
    profile=schema["execution_profile_id"] if schema["execution_profile_kind"]=="declared_physics" else None
    rebuilt = build_feature_schema(schema["endpoint"], schema["encoding_fit_id"], schema["selected_features"], supports,
                                   execution_profile_id=profile)
    if rebuilt != schema:
        raise ValueError("Feature schema identity/order differs from its declared supports")
    return schema


def validate_task(task):
    if not isinstance(task, dict) or set(task) != TASK_FIELDS or type(task.get("schema_version")) is not int or task["schema_version"] != 1:
        raise ValueError("Unsupported execution task schema")
    for key in ("task_id", "molecule_id", "encoding_fit_id", "feature_schema_id", "bundle_id", "inputs_sha256", "phase_settings_sha256", "deployment_family_id"):
        _digest(task[key], key)
    _text(task["endpoint"], "endpoint")
    for key in ("num_system_qubits", "num_index_qubits", "shots_requested"):
        positive_integer(task[key], key)
    if task["shots_requested"] > MAX_SHOTS:
        raise ValueError("Requested shot budget exceeded")
    if finite_real(task["lambda_value"], "lambda_value") <= 0:
        raise ValueError("Task lambda must be positive")
    for key in ("source_row_positions", "source_record_ids"):
        if not isinstance(task[key], list):
            raise ValueError("Task source provenance must be an explicit list")
    for position in task["source_row_positions"]:
        positive_integer(position, "source row", allow_zero=True)
    canonical_json(task)
    if _hash({k:v for k,v in task.items() if k != "task_id"}) != task["task_id"]:
        raise ValueError("Task identity differs from its contents")
    return task


def bind_job(task, job_id):
    """Persist this receipt immediately after create, before supplying inputs."""
    validate_task(task)
    binding = {"schema_version": 1, "task": task, "job_id": _text(job_id, "job_id")}
    binding["job_binding_id"] = _hash(binding)
    return binding


def validate_job_binding(binding):
    if (not isinstance(binding, dict) or set(binding) != {"schema_version", "task", "job_id", "job_binding_id"}
            or type(binding["schema_version"]) is not int or binding["schema_version"] != 1):
        raise ValueError("Invalid job binding receipt")
    validate_task(binding["task"]); _text(binding["job_id"], "job_id")
    if _hash({k:v for k,v in binding.items() if k != "job_binding_id"}) != binding["job_binding_id"]:
        raise ValueError("Job binding identity differs from its contents")
    return binding


def validate_job_bindings(bindings):
    if not isinstance(bindings,list) or not bindings:
        raise ValueError("Job bindings must be a nonempty explicit list")
    jobs, tasks = set(), set()
    for binding in bindings:
        validate_job_binding(binding)
        job, task = binding["job_id"], binding["task"]["task_id"]
        if job in jobs or task in tasks:
            raise ValueError("Duplicate job/task association in batch receipts")
        jobs.add(job); tasks.add(task)
    return True


def make_result_envelope(binding, counts, *, source):
    """Build the local normalized contract after explicit native-output mapping."""
    validate_job_binding(binding)
    if source not in SOURCES:
        raise ValueError("Result source must declare hardware, simulator, or synthetic_test")
    task = binding["task"]
    return {"schema_version": 1, "job_binding_id": binding["job_binding_id"], "job_id": binding["job_id"],
        **{k:task[k] for k in ("task_id", "molecule_id", "endpoint", "encoding_fit_id", "feature_schema_id", "bundle_id", "inputs_sha256", "shots_requested")},
        "source": source, "counts": counts}


def _bit(value):
    value = finite_real(value, "measurement bit")
    if value not in (0., 1.):
        raise ValueError("Measurements must be binary values")
    return int(value)


def _histogram_moments(histogram, supports):
    n = sum(row["count"] for row in histogram)
    result = {}
    for support in supports:
        support = tuple(support)
        result[support] = math.fsum(row["count"] * (-1 if sum(row["system_bits"][q] for q in support)%2 else 1)
                                  for row in histogram)/n
    return result


def _features_and_diagnostics(histogram, schema):
    supports = [tuple(s) for s in schema["readout_supports"]]
    needed = set(supports)
    for support in supports:
        if len(support) == 3:
            needed.update(combinations(support, 2))
    moments = _histogram_moments(histogram, needed)
    diagnostics = {"connected_pairs": [], "connected_triads": [],
                   "estimator": "empirical plug-in connected moments; finite-shot unbiasedness is not claimed"}
    for support in supports:
        if len(support) == 2:
            i,j = support
            value = moments[support]-moments[(i,)]*moments[(j,)]
            diagnostics["connected_pairs"].append({"wires": list(support), "value": value})
        if len(support) == 3:
            i,j,k = support
            value = (moments[support]-moments[(i,)]*moments[(j,k)]-moments[(j,)]*moments[(i,k)]
                     -moments[(k,)]*moments[(i,j)]+2*moments[(i,)]*moments[(j,)]*moments[(k,)])
            diagnostics["connected_triads"].append({"wires": list(support), "value": value})
    return [moments[s] for s in supports], diagnostics


def decode_result(binding, envelope, feature_schema, *, min_heralded_shots=32, require_complete=True):
    """Validate every shot and condition Z statistics on successful heralding."""
    validate_job_binding(binding); validate_feature_schema(feature_schema)
    minimum = positive_integer(min_heralded_shots, "min_heralded_shots")
    if not isinstance(require_complete, bool):
        raise ValueError("require_complete must be Boolean")
    task = binding["task"]
    if task["feature_schema_id"] != feature_schema["feature_schema_id"] or task["encoding_fit_id"] != feature_schema["encoding_fit_id"] or task["endpoint"] != feature_schema["endpoint"]:
        raise ValueError("Task and feature schema differ")
    if task["num_system_qubits"] != len(feature_schema["selected_features"]):
        raise ValueError("Task system dimension differs from feature schema")
    expected = make_result_envelope(binding, [], source=envelope.get("source") if isinstance(envelope, dict) else None)
    if not isinstance(envelope, dict) or set(envelope) != set(expected) or type(envelope["schema_version"]) is not int:
        raise ValueError("Unsupported normalized result envelope")
    if envelope["source"] != "synthetic_test" and feature_schema["execution_profile_kind"] != "declared_physics":
        raise ValueError("Hardware/simulator results require a declared physical feature-map profile")
    for key in expected:
        if key != "counts" and envelope[key] != expected[key]:
            raise ValueError(f"Result identity mismatch: {key}")
    rows = envelope["counts"]
    if not isinstance(rows, list) or not rows or len(rows) > MAX_RESULT_ROWS:
        raise ValueError("Result count rows must be a bounded nonempty list")
    received, heralded, histogram = 0, 0, Counter()
    nq, ni = task["num_system_qubits"], task["num_index_qubits"]
    for entry in rows:
        if not isinstance(entry, dict) or set(entry) != {"outputs", "count"}:
            raise ValueError("Each result count requires outputs and count")
        count = positive_integer(entry["count"], "shot count")
        values = entry["outputs"]
        if not isinstance(values, (list, tuple)) or len(values) != 3+ni+nq:
            raise ValueError("QBE result length differs from branch/real/lambda/index/system layout")
        branch, real = _bit(values[0]), _bit(values[1])
        lam = finite_real(values[2], "output lambda")
        if lam <= 0 or not math.isclose(lam, task["lambda_value"], rel_tol=1e-10):
            raise ValueError("Output lambda differs from the requested Hamiltonian")
        indices = [_bit(v) for v in values[3:3+ni]]
        system = tuple(_bit(v) for v in values[3+ni:])
        received += count
        if received > task["shots_requested"] or received > MAX_SHOTS:
            raise ValueError("Received count exceeds the requested shot budget")
        if not branch and not real and not any(indices):
            heralded += count; histogram[system] += count
    if require_complete and received != task["shots_requested"]:
        raise ValueError("Incomplete shot response; explicit partial-shot policy is required")
    if heralded < minimum:
        raise ValueError("Insufficient heralded shots for correlation features")
    hist = [{"system_bits": list(bits), "count": count} for bits,count in sorted(histogram.items())]
    values, diagnostics = _features_and_diagnostics(hist, feature_schema)
    record = {"schema_version": 1, **{k:task[k] for k in ("task_id", "molecule_id", "endpoint", "encoding_fit_id", "feature_schema_id", "bundle_id", "inputs_sha256", "shots_requested")},
        "job_id": binding["job_id"], "job_binding_id": binding["job_binding_id"], "source": envelope["source"],
        "shots_received": received, "shots_heralded": heralded, "heralding_success_rate": heralded/received,
        "min_heralded_shots": minimum, "require_complete": require_complete,
        "feature_names": list(feature_schema["feature_names"]), "feature_values": values,
        "heralded_system_counts": hist, "connected_diagnostics": diagnostics,
        "raw_moment_standard_error_bound": 1/math.sqrt(heralded),
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "shot_result_sha256": _hash(envelope)}
    record["feature_record_sha256"] = _hash(record)
    validate_feature_record(record, feature_schema, task=task)
    return record


def validate_feature_record(record, feature_schema, *, task=None):
    """Recompute features from retained histograms; a checksum is not an attestation."""
    validate_feature_schema(feature_schema)
    required = {"schema_version", "task_id", "molecule_id", "endpoint", "encoding_fit_id", "feature_schema_id", "bundle_id", "inputs_sha256", "shots_requested", "job_id", "job_binding_id", "source", "shots_received", "shots_heralded", "heralding_success_rate", "min_heralded_shots", "require_complete", "feature_names", "feature_values", "heralded_system_counts", "connected_diagnostics", "raw_moment_standard_error_bound", "uncertainty_scope", "shot_result_sha256", "feature_record_sha256"}
    if not isinstance(record, dict) or set(record) != required or type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported correlation feature record")
    for name in ("task_id", "molecule_id", "encoding_fit_id", "feature_schema_id", "bundle_id", "inputs_sha256", "job_binding_id", "shot_result_sha256", "feature_record_sha256"):
        _digest(record[name], name)
    if record["source"] not in SOURCES:
        raise ValueError("Unknown feature-record source")
    if record["uncertainty_scope"] != UNCERTAINTY_SCOPE:
        raise ValueError("Unsupported raw-moment uncertainty interpretation")
    _text(record["job_id"], "job_id")
    if record["endpoint"] != feature_schema["endpoint"] or record["encoding_fit_id"] != feature_schema["encoding_fit_id"] or record["feature_schema_id"] != feature_schema["feature_schema_id"]:
        raise ValueError("Feature record belongs to a different endpoint or fitted encoding")
    requested, received, heralded, minimum = [positive_integer(record[k], k) for k in ("shots_requested", "shots_received", "shots_heralded", "min_heralded_shots")]
    if not minimum <= heralded <= received <= requested <= MAX_SHOTS or not isinstance(record["require_complete"], bool):
        raise ValueError("Invalid feature-record shot policy/counts")
    if record["require_complete"] and received != requested:
        raise ValueError("Feature record violates its complete-shot policy")
    histogram = record["heralded_system_counts"]
    if not isinstance(histogram, list) or not histogram or len(histogram) > MAX_RESULT_ROWS:
        raise ValueError("Feature record requires its bounded heralded histogram")
    canonical = []
    for row in histogram:
        if not isinstance(row, dict) or set(row) != {"system_bits", "count"}:
            raise ValueError("Malformed heralded histogram")
        if not isinstance(row["system_bits"], list) or len(row["system_bits"]) != len(feature_schema["selected_features"]):
            raise ValueError("Histogram system dimension differs from fitted schema")
        bits = [_bit(v) for v in row["system_bits"]]
        count = positive_integer(row["count"], "heralded histogram count")
        canonical.append({"system_bits": bits, "count": count})
    if canonical != sorted(canonical, key=lambda row: row["system_bits"]) or len({tuple(r["system_bits"]) for r in canonical}) != len(canonical) or sum(r["count"] for r in canonical) != heralded:
        raise ValueError("Heralded histogram is not unique, ordered, and count-consistent")
    values, diagnostic = _features_and_diagnostics(canonical, feature_schema)
    if record["feature_names"] != feature_schema["feature_names"] or not isinstance(record["feature_values"], list) or len(record["feature_values"]) != len(values):
        raise ValueError("Feature columns differ from the fitted readout schema")
    actual = [finite_real(v, "correlation feature") for v in record["feature_values"]]
    if any(abs(a-b)>1e-12 for a,b in zip(actual, values)) or record["connected_diagnostics"] != diagnostic:
        raise ValueError("Reported correlations differ from the heralded counts")
    if not math.isclose(finite_real(record["heralding_success_rate"]), heralded/received, rel_tol=1e-12) or not math.isclose(finite_real(record["raw_moment_standard_error_bound"]),1/math.sqrt(heralded),rel_tol=1e-12):
        raise ValueError("Feature-record shot diagnostics are inconsistent")
    if _hash({k:v for k,v in record.items() if k != "feature_record_sha256"}) != record["feature_record_sha256"]:
        raise ValueError("Feature-record checksum differs from its contents")
    if task is not None:
        validate_task(task)
        for key in ("task_id", "molecule_id", "endpoint", "encoding_fit_id", "feature_schema_id", "bundle_id", "inputs_sha256", "shots_requested"):
            if record[key] != task[key]:
                raise ValueError(f"Feature record/task binding mismatch: {key}")
    return record


# Public name used by the post-API ML layer.
validate_correlation_record = validate_feature_record


def _path(value, path):
    if not isinstance(path, list) or not path or len(path)>16:
        raise ValueError("Native output paths must be explicit bounded key/index lists")
    try:
        for part in path:
            if isinstance(part, str) and isinstance(value, dict): value = value[part]
            elif isinstance(part, int) and not isinstance(part,bool) and part>=0 and isinstance(value,list): value=value[part]
            else: raise ValueError("Native output path type differs from response")
    except (KeyError, IndexError) as exc:
        raise ValueError("Native output path is absent from response") from exc
    return value


def adapt_vendor_results(response, binding, output_mapping, *, source):
    """Map an explicitly declared list of native shot objects to local rows.

    Only the SDK's outer job_id/status/results fields are established publicly.
    The deployment-specific inner mapping must be independently verified before
    hardware/simulator data are accepted. Synthetic tests remain explicit.
    """
    validate_job_binding(binding)
    if not isinstance(response, dict) or response.get("job_id") != binding["job_id"] or response.get("status") != "successful":
        raise ValueError("Vendor result job/status differs from the request receipt")
    fields = {"schema_version", "deployment_family_id", "verified", "row_path", "output_paths"}
    if not isinstance(output_mapping, dict) or set(output_mapping)!=fields or type(output_mapping["schema_version"]) is not int or output_mapping["schema_version"]!=1:
        raise ValueError("Explicit native output mapping is required")
    if output_mapping["deployment_family_id"] != binding["task"]["deployment_family_id"] or not isinstance(output_mapping["verified"],bool):
        raise ValueError("Native output mapping belongs to another deployment family")
    if source != "synthetic_test" and output_mapping["verified"] is not True:
        raise ValueError("Native output mapping remains unverified")
    paths = output_mapping["output_paths"]
    task=binding["task"]
    if not isinstance(paths,list) or len(paths)!=3+task["num_index_qubits"]+task["num_system_qubits"]:
        raise ValueError("Native output mapping dimensions differ from the task")
    rows = _path(response, output_mapping["row_path"])
    if not isinstance(rows,list) or not rows or len(rows)>MAX_RESULT_ROWS:
        raise ValueError("Declared native shot records must be a bounded nonempty list")
    counts=Counter()
    for row in rows:
        values=[finite_real(_path(row,p),"native result value") for p in paths]
        counts[tuple(values)]+=1
    normalized=[{"outputs":list(values),"count":count} for values,count in sorted(counts.items())]
    return make_result_envelope(binding, normalized, source=source)
