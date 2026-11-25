"""All-molecule preparation and dense-reference post-API learning integration.

Shot records here come from the independently assembled coherent QSVT model,
not a Triple Alpha call. Invented labels exercise supervised software only.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"python_classes"), str(ROOT/"tests")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"admet-test-matplotlib"))
from admet_curation import CurationConfig
from admet_submission_entry import _strict_json
from batch_execution import BatchSettings, validate_batch_manifest
from learning_workflow import prepare_learning_workflow, load_preparation, train_post_api
from native_binding import named_inputs, render_native_main
from post_api_ml import ReadoutConfig, read_model
from quantum_reference import lcu_operators, heralded_evolution_unitary
from result_contract import (bind_job, make_result_envelope, decode_result, validate_feature_record,
                             adapt_vendor_results, build_feature_schema)
from runtime_input_protocol import canonical_json
from test_post_api_ml import SMILES


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class BatchAndResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name)
        table = pd.DataFrame({"Drug":SMILES, "Y":[i%2 for i in range(len(SMILES))]})
        # One consistent duplicate and one contradictory observation. Both
        # structures remain represented; the conflicting label is quarantined.
        table = pd.concat([table,pd.DataFrame({"Drug":[SMILES[0],SMILES[3]],"Y":[0,0]})],ignore_index=True)
        cls.directory, cls.summary = prepare_learning_workflow(table, "fixture", cls.output,
            curation_config=CurationConfig(top_k=2,n_bits=32),
            batch_settings=BatchSettings(shots_requested=32768),
            readout_config=ReadoutConfig(regularization_candidates=(1.,),minimum_heralded_shots=64))
        cls.documents = load_preparation(cls.directory)
        cls.batch = cls.documents["batch_manifest.json"]
        cls.schema = cls.batch["feature_schema"]
        cls.bindings = [bind_job(task,"synthetic-"+task["task_id"]) for task in cls.batch["tasks"]]
        cls.envelopes = []
        probability_cache = {}
        for binding in cls.bindings:
            task = binding["task"]
            if task["bundle_id"] not in probability_cache:
                bundle = cls.output/"batch"/"bundles"/task["bundle_id"]
                payload = _strict_json((bundle/"hamiltonian.json").read_text())
                settings = _strict_json((bundle/"qsvt_settings.json").read_text())
                operators = lcu_operators(payload,max_dimension=256)
                U = heralded_evolution_unitary(operators,settings,max_dimension=1024)
                initial = np.ones(1,complex)
                for theta in payload["feature_vector"]:
                    initial = np.kron(initial,[np.cos(np.pi*theta/2),np.sin(np.pi*theta/2)])
                state = np.zeros(len(U),complex);state[:len(initial)] = initial
                probabilities = np.abs(U@state)**2
                probability_cache[task["bundle_id"]] = probabilities/probabilities.sum()
            rng = np.random.default_rng(int(task["molecule_id"][:8],16))
            observed = rng.multinomial(task["shots_requested"],probability_cache[task["bundle_id"]])
            width = 2+task["num_index_qubits"]+task["num_system_qubits"]
            counts = []
            for index,count in enumerate(observed):
                if count:
                    bits = [int(x) for x in format(index,f"0{width}b")]
                    counts.append({"outputs":bits[:2]+[task["lambda_value"]]+bits[2:],"count":int(count)})
            cls.envelopes.append(make_result_envelope(binding,counts,source="synthetic_test"))
        cls.results = {"schema_version":1,"batch_id":cls.batch["batch_id"],
                       "bindings":cls.bindings,"results":cls.envelopes}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_every_structure_and_raw_row_is_represented_without_label_leakage(self):
        self.assertEqual(self.summary["raw_rows"],22)
        self.assertEqual(self.summary["cleaned_labelled_identities"],19)
        self.assertEqual(self.summary["all_valid_inference_identities"],20)
        self.assertEqual(self.summary["prepared_molecules"],20)
        self.assertEqual(sum(len(t["source_row_positions"]) for t in self.batch["tasks"]),22)
        self.assertFalse(self.summary["quantum_model_fitted"])
        self.assertIsNone(self.summary["quantum_prediction_metrics"])
        self.assertTrue(all("Y" not in t for t in self.batch["tasks"]))
        self.assertEqual(self.summary["unique_physics"]+self.summary["physics_reuse_count"],20)

    def test_native_names_roundtrip_every_logical_scalar_and_output_shape(self):
        bindings = self.documents["native_bindings.json"]
        for group in self.batch["physics_groups"]:
            native = next(b for b in bindings if b["deployment_family_id"]==group["deployment_family"]["deployment_family_id"])
            document = _strict_json((self.output/"batch"/"bundles"/group["bundle_id"]/"runtime_inputs.json").read_text())
            pairs = named_inputs(document,native)
            self.assertEqual([float(v) for _,v in pairs],document["inputs"][5:])
            self.assertEqual(len(pairs),len({name for name,_ in pairs}))
            self.assertFalse(native["native_compiler_verified"])
            self.assertEqual(len(native["output_names"]),3+group["num_index_qubits"]+group["num_system_qubits"])
            source = render_native_main(group["deployment_family"])
            self.assertEqual(source.count("input -> num "),len(pairs))

    def test_heralded_decoder_matches_direct_parity_from_joint_histogram(self):
        record = decode_result(self.bindings[0],self.envelopes[0],self.schema,min_heralded_shots=64)
        n = record["shots_heralded"]
        for support,value in zip(self.schema["readout_supports"],record["feature_values"]):
            expected = sum(r["count"]*(-1)**sum(r["system_bits"][q] for q in support)
                           for r in record["heralded_system_counts"])/n
            self.assertAlmostEqual(value,expected)
        self.assertLess(record["raw_moment_standard_error_bound"],.1)

    def test_result_identity_bits_lambda_and_shot_failures_reject(self):
        cases = []
        wrong = copy.deepcopy(self.envelopes[0]);wrong["bundle_id"]="0"*64;cases.append(wrong)
        wrong = copy.deepcopy(self.envelopes[0]);wrong["counts"][0]["outputs"][2]*=2;cases.append(wrong)
        wrong = copy.deepcopy(self.envelopes[0]);wrong["counts"][0]["outputs"][0]=2;cases.append(wrong)
        wrong = copy.deepcopy(self.envelopes[0]);wrong["counts"][0]["count"]+=1;cases.append(wrong)
        for envelope in cases:
            with self.assertRaises(ValueError):decode_result(self.bindings[0],envelope,self.schema)
        record = decode_result(self.bindings[0],self.envelopes[0],self.schema)
        record["feature_values"][0] = -record["feature_values"][0]+.1
        record["feature_record_sha256"] = digest({k:v for k,v in record.items() if k!="feature_record_sha256"})
        with self.assertRaises(ValueError):validate_feature_record(record,self.schema)

    def test_batch_family_and_physics_mutations_fail_even_when_rehashed(self):
        for mutate in (lambda m:m["physics_groups"][0].__setitem__("molecule_count",999),
                       lambda m:m["deployment_families"][0].__setitem__("native_capacity_verified",True),
                       lambda m:m["tasks"][0].__setitem__("deployment_family_id","0"*64)):
            changed = copy.deepcopy(self.batch);mutate(changed)
            changed["batch_id"] = digest({k:v for k,v in changed.items() if k!="batch_id"})
            with self.assertRaises(ValueError):validate_batch_manifest(changed,output_root=self.output/"batch")

    def test_post_api_model_fits_roundtrips_and_predicts_quarantined_identity(self):
        model,report = train_post_api(self.directory,self.results,self.output/"trained",
            config=ReadoutConfig(regularization_candidates=(1.,),minimum_heralded_shots=64))
        self.assertEqual(model["data_source"],"synthetic_test")
        self.assertEqual(report["prediction_count"],20)
        self.assertEqual(read_model(self.output/"trained"/"correlation_model.json"),model)
        self.assertEqual(report["test_evaluation"]["evaluation_split"],"untouched_test")
        self.assertEqual(report["triple_alpha_calls"],0)
        changed = copy.deepcopy(self.results);changed["results"].pop()
        with self.assertRaises(ValueError):train_post_api(self.directory,changed,self.output/"missing")
        changed = copy.deepcopy(self.results);changed["results"][0]=None
        with self.assertRaises(ValueError):train_post_api(self.directory,changed,self.output/"malformed")

    def test_execution_profile_changes_measurement_feature_identity(self):
        changed=copy.deepcopy(self.batch)
        changed["settings"]["physical_time"]*=2
        changed["batch_id"]=digest({k:v for k,v in changed.items() if k!="batch_id"})
        with self.assertRaisesRegex(ValueError,"profile"):
            validate_batch_manifest(changed)
        schema=build_feature_schema(self.schema["endpoint"],self.schema["encoding_fit_id"],
            self.schema["selected_features"],self.documents["encoding_schema.json"]["readout_supports"],
            execution_profile_id="0"*64)
        with self.assertRaises(ValueError):decode_result(self.bindings[0],self.envelopes[0],schema)

    def test_native_result_mapping_requires_explicit_verified_deployment_paths(self):
        task = self.bindings[0]["task"]
        sample = self.envelopes[0]["counts"][0]["outputs"]
        response = {"job_id":self.bindings[0]["job_id"],"status":"successful","results":[{"values":sample}]}
        mapping = {"schema_version":1,"deployment_family_id":task["deployment_family_id"],"verified":False,
                   "row_path":["results"],"output_paths":[["values",i] for i in range(len(sample))]}
        with self.assertRaises(ValueError):adapt_vendor_results(response,self.bindings[0],mapping,source="hardware")
        envelope = adapt_vendor_results(response,self.bindings[0],mapping,source="synthetic_test")
        self.assertEqual(envelope["source"],"synthetic_test")
