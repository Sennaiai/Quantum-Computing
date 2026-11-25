"""Portable supervised readouts for validated post-API Z-correlation records.

Real quantum models cannot be fitted before real result records exist. The same
code is exercised with explicitly labelled synthetic/reference records in tests;
those results are never relabelled as hardware or ADMET prediction evidence.
The frozen molecular encoding and split are fitted on training molecules only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers
from pathlib import Path
import re
import sys
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, log_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python_classes") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_classes"))
from runtime_input_protocol import atomic_write_json, canonical_json

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_CELLS = 30_000_000
MAX_ROWS = 50_000
MAX_FEATURES = 65_536


@dataclass(frozen=True)
class ReadoutConfig:
    regularization_candidates: tuple[float, ...] = (.1, 1., 10.)
    class_weight: str | None = None
    max_iterations: int = 5000
    tolerance: float = 1e-9
    minimum_heralded_shots: int = 256

    def validate(self):
        values = self.regularization_candidates
        if (not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 16
                or any(isinstance(x, bool) or not isinstance(x, numbers.Real)
                       or not math.isfinite(x) or x <= 0 for x in values)):
            raise ValueError("Regularization candidates must be bounded positive finite real values")
        if list(values) != sorted(set(values)):
            raise ValueError("Regularization candidates must be unique and increasing")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or balanced")
        for name, minimum, maximum in (("max_iterations", 10, 100000),
                                        ("minimum_heralded_shots", 1, 1000000000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is outside its integer budget")
        if (isinstance(self.tolerance, bool) or not isinstance(self.tolerance, numbers.Real)
                or not math.isfinite(self.tolerance) or not 1e-12 <= self.tolerance <= 1e-3):
            raise ValueError("tolerance must lie in [1e-12, 1e-3]")


def _configuration(config):
    config = ReadoutConfig() if config is None else config
    if not isinstance(config, ReadoutConfig):
        raise ValueError("config must be a ReadoutConfig")
    config.validate()
    return config


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _ids(values, name):
    if (not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_ROWS
            or any(not isinstance(x, str) or not x or x != x.strip() for x in values)):
        raise ValueError(f"{name} must be nonempty molecule identifiers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate molecule identifiers")
    return list(values)


def _hash(value, name):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 identifier")
    return value


def _partition_contract(encoding_schema, split_manifest):
    if not isinstance(encoding_schema, dict) or not isinstance(split_manifest, dict):
        raise ValueError("Frozen encoding and split manifest must be objects")
    from data_quality import validate_split_manifest
    from supervised_encoding import validate_frozen_schema
    validate_frozen_schema(encoding_schema)
    validate_split_manifest(split_manifest)
    endpoint = encoding_schema.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint or endpoint != split_manifest.get("endpoint"):
        raise ValueError("Encoding and split endpoint differ")
    encoding_id = _hash(encoding_schema.get("encoding_fit_id"), "encoding_fit_id")
    if encoding_schema.get("schema_id") != encoding_id or encoding_schema.get("fit_scope") != "training_only":
        raise ValueError("ML requires a training-only frozen encoding")
    partitions = {name: _ids(split_manifest.get(name), f"split {name}") for name in ("train", "validation", "test")}
    if any(set(partitions[a]) & set(partitions[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Molecule identities overlap supervised partitions")
    if set(_ids(encoding_schema.get("train_molecule_ids"), "encoding train IDs")) != set(partitions["train"]):
        raise ValueError("Encoding was not fitted on exactly the declared training partition")
    groups = split_manifest.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("Split manifest must declare scaffold groups")
    group_sets = {}
    for name in partitions:
        group_sets[name] = set(_ids(groups.get(name), f"{name} scaffold groups"))
    if any(group_sets[a] & group_sets[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Scaffold groups overlap supervised partitions")
    split_id = _hash(split_manifest.get("split_id"), "split_id")
    return endpoint, encoding_id, split_id, partitions


def _matrix(values, name):
    if isinstance(values, np.ndarray):
        shape = values.shape
        if values.ndim != 2:
            raise ValueError(f"{name} must be a rectangular numeric matrix")
    elif isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple, np.ndarray)):
        shape = (len(values), len(values[0]))
        if any(not isinstance(row, (list, tuple, np.ndarray)) or len(row) != shape[1] for row in values):
            raise ValueError(f"{name} must be rectangular")
    else:
        raise ValueError(f"{name} must be a rectangular numeric matrix")
    if not 1 <= shape[0] <= MAX_ROWS or not 1 <= shape[1] <= MAX_FEATURES or shape[0]*shape[1] > MAX_CELLS:
        raise ValueError(f"{name} exceeds the bounded readout allocation")
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _labels(labels, ids, name):
    if not isinstance(labels, dict) or set(labels) != set(ids):
        raise ValueError(f"{name} must map exactly the supplied molecule IDs to labels")
    values = [labels[i] for i in ids]
    if any(isinstance(x, complex) or not isinstance(x, (numbers.Real, np.bool_)) or not math.isfinite(x) or x not in (0,1) for x in values):
        raise ValueError(f"{name} must contain binary real labels")
    result = np.asarray(values, dtype=int)
    if set(result.tolist()) != {0,1}:
        raise ValueError(f"{name} requires both binary classes")
    return result


def binary_metrics(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    if (y.ndim != 1 or p.shape != y.shape or len(y) < 2 or y.dtype.kind not in "biuf"
            or np.iscomplexobj(y) or not np.isfinite(y).all() or not np.isin(y, [0,1]).all()
            or set(y.tolist()) != {0,1} or p.dtype.kind not in "biuf" or np.iscomplexobj(p)
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Metrics require aligned finite binary labels and probabilities with both classes")
    prediction = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0,1]).ravel()
    return {"samples": len(y), "positive_prevalence": float(np.mean(y)),
            "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0,1])),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, prediction)),
            "confusion_at_0_5": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def _bind_training_labels(labels, encoding_schema, ids):
    _labels(labels, ids, "training labels")
    digest = _digest([[identity, int(labels[identity])] for identity in sorted(ids)])
    if digest != encoding_schema["training_labels_sha256"]:
        raise ValueError("Training labels differ from the frozen Hamiltonian fit; refit the encoding for changed labels")
    return digest


def _feature_names(names, width):
    if (not isinstance(names, (list, tuple)) or len(names) != width
            or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != width):
        raise ValueError("Feature names must uniquely describe every matrix column")
    return list(names)


def _fit_numeric(train_X, train_labels, train_ids, validation_X, validation_labels, validation_ids,
                 *, feature_names, feature_schema_id, endpoint, encoding_id, split_id, partitions,
                 source, kind, config):
    config.validate()
    train_X, validation_X = _matrix(train_X, "training features"), _matrix(validation_X, "validation features")
    train_ids, validation_ids = _ids(train_ids, "training IDs"), _ids(validation_ids, "validation IDs")
    if (len(train_ids) != len(train_X) or len(validation_ids) != len(validation_X)
            or set(train_ids) != set(partitions["train"]) or set(validation_ids) != set(partitions["validation"])):
        raise ValueError("Feature records must completely cover their declared training/validation partitions")
    if train_X.shape[1] != validation_X.shape[1]:
        raise ValueError("Training and validation feature dimensions differ")
    names = _feature_names(feature_names, train_X.shape[1])
    y_train, y_validation = _labels(train_labels, train_ids, "training labels"), _labels(validation_labels, validation_ids, "validation labels")
    if min(np.bincount(y_train, minlength=2)) < 2:
        raise ValueError("Training requires at least two molecules in each class")
    scaler = StandardScaler().fit(train_X)
    transformed, validation_transformed = scaler.transform(train_X), scaler.transform(validation_X)
    trials, best = [], None
    for C in config.regularization_candidates:
        classifier = LogisticRegression(C=float(C), class_weight=config.class_weight, solver="lbfgs",
            max_iter=config.max_iterations, tol=config.tolerance, random_state=42)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(transformed, y_train)
        except ConvergenceWarning:
            raise ValueError("Logistic readout did not converge; model was not accepted") from None
        metrics = binary_metrics(y_validation, classifier.predict_proba(validation_transformed)[:,1])
        trials.append({"C": float(C), "metrics": metrics})
        # Stable tie policy: sorted candidates retain the smaller C.
        if best is None or metrics["roc_auc"] > best[0]:
            best = (metrics["roc_auc"], classifier, float(C))
    classifier = best[1]
    coefficients = classifier.coef_[0].astype(float)
    if not np.isfinite(coefficients).all() or not np.isfinite(classifier.intercept_).all():
        raise ValueError("Fitted model contains nonfinite coefficients")
    artifact = {"model_schema_version": 1, "status": "fitted", "model_type": "standardized_binary_logistic",
        "kind": kind, "endpoint": endpoint, "encoding_fit_id": encoding_id, "split_id": split_id,
        "feature_schema_id": _hash(feature_schema_id, "feature_schema_id"), "feature_names": names,
        "data_source": source, "classes": [0,1], "threshold": .5,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficient": coefficients.tolist(), "intercept": float(classifier.intercept_[0]),
        "selected_C": best[2], "config": json.loads(canonical_json(asdict(config))),
        "train_molecule_ids": train_ids, "validation_molecule_ids": validation_ids,
        "test_molecule_ids": list(partitions["test"]),
        "training_label_sha256": _digest([[i, int(train_labels[i])] for i in sorted(train_ids)]),
        "validation_label_sha256": _digest([[i, int(validation_labels[i])] for i in sorted(validation_ids)]),
        "training_class_counts": {str(i): int(np.sum(y_train == i)) for i in (0,1)},
        "training_positive_prevalence": float(np.mean(y_train)),
        "validation_selection": {"metric": "roc_auc", "tie_policy": "smaller C", "trials": trials},
        "probability_policy": "sigmoid readout; no separate calibration or predictive interval claim",
        "test_labels_used_in_fit": False}
    artifact["model_id"] = _digest(artifact)
    validate_model(artifact)
    # Verify the exported arithmetic against the fitted sklearn estimator.
    exported = _numeric_predict(artifact, validation_X)
    if not np.allclose(exported, classifier.predict_proba(validation_transformed)[:,1], rtol=1e-12, atol=1e-14):
        raise ValueError("Portable logistic arithmetic differs from the fitted estimator")
    return artifact


def validate_model(model):
    if not isinstance(model, dict) or type(model.get("model_schema_version")) is not int or model["model_schema_version"] != 1:
        raise ValueError("Unsupported readout model schema")
    expected = {"model_schema_version", "status", "model_type", "kind", "endpoint", "encoding_fit_id", "split_id",
        "feature_schema_id", "feature_names", "data_source", "classes", "threshold", "scaler_mean", "scaler_scale",
        "coefficient", "intercept", "selected_C", "config", "train_molecule_ids", "validation_molecule_ids",
        "test_molecule_ids", "training_label_sha256", "validation_label_sha256", "training_class_counts", "training_positive_prevalence",
        "validation_selection", "probability_policy", "test_labels_used_in_fit", "model_id"}
    if set(model) != expected:
        raise ValueError("Readout model fields differ from schema")
    body = {key:value for key,value in model.items() if key != "model_id"}
    if _hash(model.get("model_id"), "model_id") != _digest(body):
        raise ValueError("Readout model checksum differs from its content")
    if (model.get("status") != "fitted" or model.get("model_type") != "standardized_binary_logistic"
            or model.get("kind") not in {"quantum_z_correlations", "classical_full_counts", "classical_selected_counts"}
            or model.get("classes") != [0,1] or any(type(x) is not int for x in model.get("classes", []))
            or model.get("threshold") != .5
            or model.get("test_labels_used_in_fit") is not False):
        raise ValueError("Invalid readout model policy")
    for name in ("encoding_fit_id", "split_id", "feature_schema_id"):
        _hash(model.get(name), name)
    if not isinstance(model.get("endpoint"), str) or not model["endpoint"] or not isinstance(model.get("data_source"), str):
        raise ValueError("Invalid readout endpoint/source")
    names = model.get("feature_names")
    if not isinstance(names, list):
        raise ValueError("Model feature names must be a list")
    _feature_names(names, len(names))
    if not 1 <= len(names) <= MAX_FEATURES:
        raise ValueError("Readout feature budget exceeded")
    for name in ("scaler_mean", "scaler_scale", "coefficient"):
        values = model.get(name)
        if (not isinstance(values, list) or len(values) != len(names)
                or any(isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) for x in values)):
            raise ValueError(f"Invalid model {name}")
    if any(value <= 0 for value in model["scaler_scale"]):
        raise ValueError("Model scales must be positive")
    if isinstance(model.get("intercept"), bool) or not isinstance(model.get("intercept"), numbers.Real) or not math.isfinite(model["intercept"]):
        raise ValueError("Model intercept must be finite")
    try:
        config = ReadoutConfig(**model["config"]); config.validate()
    except (TypeError, ValueError):
        raise ValueError("Invalid model configuration") from None
    if (isinstance(model.get("selected_C"), bool) or not isinstance(model.get("selected_C"), numbers.Real)
            or model.get("selected_C") not in config.regularization_candidates):
        raise ValueError("Selected C is absent from the declared search")
    sets = {key: set(_ids(model.get(key), key)) for key in ("train_molecule_ids", "validation_molecule_ids", "test_molecule_ids")}
    if any(sets[a] & sets[b] for a,b in (("train_molecule_ids","validation_molecule_ids"),("train_molecule_ids","test_molecule_ids"),("validation_molecule_ids","test_molecule_ids"))):
        raise ValueError("Readout model partitions overlap")
    counts = model["training_class_counts"]
    if (not isinstance(counts, dict) or set(counts) != {"0", "1"}
            or any(type(x) is not int or x < 2 for x in counts.values())
            or sum(counts.values()) != len(sets["train_molecule_ids"])):
        raise ValueError("Invalid model training class counts")
    prevalence = model["training_positive_prevalence"]
    if (isinstance(prevalence, bool) or not isinstance(prevalence, numbers.Real)
            or not math.isfinite(prevalence) or not math.isclose(prevalence, counts["1"]/sum(counts.values()), rel_tol=1e-12)):
        raise ValueError("Training prevalence differs from class counts")
    _hash(model["training_label_sha256"], "training_label_sha256")
    _hash(model["validation_label_sha256"], "validation_label_sha256")
    selection = model["validation_selection"]
    if (not isinstance(selection, dict) or set(selection) != {"metric", "tie_policy", "trials"}
            or selection["metric"] != "roc_auc" or selection["tie_policy"] != "smaller C"
            or not isinstance(selection["trials"], list) or len(selection["trials"]) != len(config.regularization_candidates)):
        raise ValueError("Invalid validation-only selection metadata")
    values = []
    for trial, C in zip(selection["trials"], config.regularization_candidates):
        if not isinstance(trial, dict) or set(trial) != {"C", "metrics"} or trial["C"] != C or not isinstance(trial["metrics"], dict):
            raise ValueError("Validation trial differs from the declared regularization search")
        score = trial["metrics"].get("roc_auc")
        if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation ROC-AUC")
        values.append(score)
    if model["selected_C"] != config.regularization_candidates[values.index(max(values))]:
        raise ValueError("Selected C disagrees with validation-only ranking")
    sources = {"hardware", "simulator", "synthetic_test"} if model["kind"] == "quantum_z_correlations" else {"classical_molecular_counts"}
    if model["data_source"] not in sources:
        raise ValueError("Model evidence source is incompatible with its type")
    return True


def _numeric_predict(model, X):
    X = _matrix(X, "prediction features")
    if X.shape[1] != len(model["feature_names"]):
        raise ValueError("Prediction feature dimension differs from model")
    values = (X-np.asarray(model["scaler_mean"]))/np.asarray(model["scaler_scale"])
    scores = values@np.asarray(model["coefficient"])+model["intercept"]
    if not np.isfinite(scores).all():
        raise ValueError("Prediction arithmetic overflowed")
    return expit(scores)


def _correlation_matrix(records, feature_schema, *, expected_ids=None, minimum_shots=256,
                        expected_source=None):
    if not isinstance(feature_schema, dict):
        raise ValueError("A validated correlation feature schema is required")
    from result_contract import validate_feature_schema, validate_feature_record
    validate_feature_schema(feature_schema)
    schema_id = _hash(feature_schema.get("feature_schema_id"), "feature_schema_id")
    names = feature_schema.get("feature_names")
    if not isinstance(names, list):
        raise ValueError("Correlation schema requires ordered feature names")
    _feature_names(names, len(names))
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Result records are missing; quantum model awaits API results")
    ids, values, sources = [], [], set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Correlation records must be objects")
        validate_feature_record(row, feature_schema)
        if row.get("feature_schema_id") != schema_id or row.get("feature_names") != names:
            raise ValueError("Correlation feature schema mismatch")
        for key in ("endpoint", "encoding_fit_id"):
            if row.get(key) != feature_schema.get(key):
                raise ValueError(f"Correlation {key} mismatch")
        _hash(row.get("bundle_id"), "bundle_id"); _hash(row.get("inputs_sha256"), "inputs_sha256")
        accepted, received, requested = row.get("shots_heralded"), row.get("shots_received"), row.get("shots_requested")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (accepted,received,requested)) or not accepted <= received <= requested:
            raise ValueError("Invalid correlation shot accounting")
        if accepted < minimum_shots:
            raise ValueError("Insufficient heralded shots for the declared readout QC")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError("Correlation result must declare its measurement source")
        sources.add(row["source"])
        ids.append(row.get("molecule_id")); values.append(row.get("feature_values"))
    ids = _ids(ids, "result molecule IDs")
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise ValueError("Results must completely cover the declared molecule partition")
    if len(sources) != 1 or (expected_source is not None and sources != {expected_source}):
        raise ValueError("Mixed or shifted measurement sources are not accepted")
    X = _matrix(values, "correlation values")
    if X.shape[1] != len(names) or np.any(np.abs(X) > 1+1e-12):
        raise ValueError("Raw Z-correlation values must lie in [-1,1]")
    return X, ids, next(iter(sources))


def _readout_encoding_contract(encoding_schema, feature_schema):
    from result_contract import build_feature_schema, validate_feature_schema
    validate_feature_schema(feature_schema)
    rebuilt = build_feature_schema(encoding_schema["endpoint"], encoding_schema["encoding_fit_id"],
        encoding_schema["active_original_features"], encoding_schema["readout_supports"],
        execution_profile_id=(feature_schema["execution_profile_id"]
            if feature_schema["execution_profile_kind"] == "declared_physics" else None))
    if rebuilt != feature_schema:
        raise ValueError("Readout feature mapping/supports differ from the frozen Hamiltonian encoding")
    return feature_schema


def prepare_readout_spec(encoding_schema, feature_schema, split_manifest, config=None):
    """Describe an unfitted quantum model without inventing outputs or metrics."""
    config = _configuration(config)
    endpoint, encoding_id, split_id, partitions = _partition_contract(encoding_schema, split_manifest)
    _readout_encoding_contract(encoding_schema, feature_schema)
    if feature_schema.get("encoding_fit_id") != encoding_id or feature_schema.get("endpoint") != endpoint:
        raise ValueError("Readout and encoding schema differ")
    names = _feature_names(feature_schema.get("feature_names"), len(feature_schema.get("feature_names", [])))
    return {"schema_version": 1, "status": "awaiting_api_results", "model_type": "standardized_binary_logistic",
            "endpoint": endpoint, "encoding_fit_id": encoding_id, "split_id": split_id,
            "feature_schema_id": _hash(feature_schema.get("feature_schema_id"), "feature_schema_id"),
            "feature_names": names, "config": json.loads(canonical_json(asdict(config))), "partition_sizes": {k:len(v) for k,v in partitions.items()},
            "fitted_coefficients": None, "quantum_prediction_metrics": None,
            "reason": "No real quantum result records have been supplied"}


def fit_correlation_model(train_records, train_labels, *, validation_records, validation_labels,
                          encoding_schema, feature_schema, split_manifest, config=None):
    """Fit on training records, select C on validation only; no test-label input."""
    config = _configuration(config)
    endpoint, encoding_id, split_id, partitions = _partition_contract(encoding_schema, split_manifest)
    _readout_encoding_contract(encoding_schema, feature_schema)
    if feature_schema.get("endpoint") != endpoint or feature_schema.get("encoding_fit_id") != encoding_id:
        raise ValueError("Readout schema differs from the frozen supervised encoding")
    if not isinstance(train_records, (list, tuple)) or not train_records:
        raise ValueError("Result records are missing; quantum model awaits API results")
    _bind_training_labels(train_labels, encoding_schema, partitions["train"])
    X, ids, source = _correlation_matrix(train_records, feature_schema, expected_ids=partitions["train"],
                                         minimum_shots=config.minimum_heralded_shots)
    V, val_ids, _ = _correlation_matrix(validation_records, feature_schema, expected_ids=partitions["validation"],
        minimum_shots=config.minimum_heralded_shots, expected_source=source)
    return _fit_numeric(X, train_labels, ids, V, validation_labels, val_ids,
        feature_names=feature_schema["feature_names"], feature_schema_id=feature_schema["feature_schema_id"],
        endpoint=endpoint, encoding_id=encoding_id, split_id=split_id, partitions=partitions,
        source=source, kind="quantum_z_correlations", config=config)


def predict_correlation_model(model, records, *, feature_schema):
    validate_model(model)
    if (not isinstance(feature_schema, dict) or model["kind"] != "quantum_z_correlations"
            or feature_schema.get("feature_schema_id") != model["feature_schema_id"]
            or feature_schema.get("encoding_fit_id") != model["encoding_fit_id"]
            or feature_schema.get("endpoint") != model["endpoint"]):
        raise ValueError("Prediction schema/model type differs")
    X, ids, _ = _correlation_matrix(records, feature_schema,
        minimum_shots=model["config"]["minimum_heralded_shots"], expected_source=model["data_source"])
    probabilities = _numeric_predict(model, X)
    return [{"molecule_id": i, "endpoint": model["endpoint"], "probability_positive": float(p),
             "prediction_at_0_5": int(p >= .5), "model_id": model["model_id"],
             "measurement_source": model["data_source"]} for i,p in zip(ids, probabilities)]


def evaluate_correlation_model(model, test_records, test_labels, *, feature_schema):
    prediction = predict_correlation_model(model, test_records, feature_schema=feature_schema)
    ids = [row["molecule_id"] for row in prediction]
    if set(ids) != set(model["test_molecule_ids"]):
        raise ValueError("Evaluation must use the complete untouched test partition")
    metrics = binary_metrics(_labels(test_labels, ids, "test labels"), [row["probability_positive"] for row in prediction])
    return {"model_id": model["model_id"], "endpoint": model["endpoint"], "evaluation_split": "untouched_test",
            "measurement_source": model["data_source"], "metrics": metrics,
            "interpretation": "Synthetic/reference results validate software only; hardware evidence requires hardware records"}


def _classical_matrix(records, encoding_schema, representation):
    if representation not in {"full_counts", "theta"}:
        raise ValueError("Classical representation must be full_counts or theta")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Molecular transform records are required")
    ids, values = [], []
    for row in records:
        if (not isinstance(row, dict) or row.get("endpoint") != encoding_schema["endpoint"]
                or row.get("encoding_fit_id") != encoding_schema["encoding_fit_id"]):
            raise ValueError("Classical molecular schema mismatch")
        ids.append(row.get("molecule_id"));values.append(row.get(representation))
    ids = _ids(ids, "molecular IDs")
    X = _matrix(values, "classical features")
    if np.any(X < 0) or (representation == "theta" and np.any(X > 1)) or (representation == "full_counts" and not np.equal(X, np.floor(X)).all()):
        raise ValueError("Classical features violate count/scale domains")
    expected = encoding_schema["config"]["n_bits"] if representation == "full_counts" else len(encoding_schema["active_original_features"])
    if X.shape[1] != expected:
        raise ValueError("Classical feature dimension differs from the frozen encoding")
    names = ([f"morgan_count[{i}]" for i in range(expected)] if representation == "full_counts" else
             [f"selected_count[{i}]" for i in encoding_schema["active_original_features"]])
    schema_id = _digest({"encoding_fit_id": encoding_schema["encoding_fit_id"], "representation": representation, "feature_names": names})
    return X, ids, names, schema_id


def fit_classical_baseline(train_records, train_labels, *, validation_records, validation_labels,
                           encoding_schema, split_manifest, representation="full_counts", config=None):
    """Matched-split classical baseline, using only training/validation labels."""
    config = _configuration(config)
    endpoint, encoding_id, split_id, partitions = _partition_contract(encoding_schema, split_manifest)
    _bind_training_labels(train_labels, encoding_schema, partitions["train"])
    X, ids, names, schema_id = _classical_matrix(train_records, encoding_schema, representation)
    V, val_ids, _, _ = _classical_matrix(validation_records, encoding_schema, representation)
    kind = "classical_full_counts" if representation == "full_counts" else "classical_selected_counts"
    return _fit_numeric(X, train_labels, ids, V, validation_labels, val_ids,
        feature_names=names, feature_schema_id=schema_id, endpoint=endpoint, encoding_id=encoding_id,
        split_id=split_id, partitions=partitions, source="classical_molecular_counts", kind=kind, config=config)


def predict_classical_baseline(model, records, *, encoding_schema):
    validate_model(model)
    from supervised_encoding import validate_frozen_schema
    validate_frozen_schema(encoding_schema)
    if model["endpoint"] != encoding_schema.get("endpoint"):
        raise ValueError("Classical endpoint differs from model")
    representation = "full_counts" if model["kind"] == "classical_full_counts" else "theta" if model["kind"] == "classical_selected_counts" else None
    if representation is None or model["encoding_fit_id"] != encoding_schema.get("encoding_fit_id"):
        raise ValueError("Classical model/encoding mismatch")
    X, ids, names, schema_id = _classical_matrix(records, encoding_schema, representation)
    if names != model["feature_names"] or schema_id != model["feature_schema_id"]:
        raise ValueError("Classical feature layout differs from model")
    return [{"molecule_id": i, "endpoint": model["endpoint"], "probability_positive": float(p),
             "prediction_at_0_5": int(p >= .5), "model_id": model["model_id"]}
            for i,p in zip(ids, _numeric_predict(model, X))]


def evaluate_classical_baseline(model, test_records, test_labels, *, encoding_schema):
    predictions = predict_classical_baseline(model, test_records, encoding_schema=encoding_schema)
    ids = [row["molecule_id"] for row in predictions]
    if set(ids) != set(model["test_molecule_ids"]):
        raise ValueError("Classical evaluation must use the identical complete test partition")
    return {"model_id": model["model_id"], "endpoint": model["endpoint"], "evaluation_split": "untouched_test",
            "representation": model["kind"], "metrics": binary_metrics(_labels(test_labels, ids, "test labels"),
                [row["probability_positive"] for row in predictions])}


def write_model(path, model):
    validate_model(model)
    atomic_write_json(Path(path), model)


def read_model(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("Model must be a bounded regular JSON file")
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:raise ValueError("Duplicate model JSON key")
            result[key]=value
        return result
    model = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    validate_model(model)
    return model
