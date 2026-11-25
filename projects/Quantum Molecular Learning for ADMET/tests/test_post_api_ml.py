"""Leakage, shot/identity contracts, and portable readout validation.

The quantum records in this module are deliberately synthetic count fixtures,
not outputs of a real API call or evidence of ADMET predictive performance.
"""
from pathlib import Path
import copy
import inspect
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'python_classes')]
from admet_curation import CurationConfig
from data_quality import clean_endpoint_table, split_cleaned_dataset, sha256_json
from supervised_encoding import fit_hamiltonian_schema, transform_molecules
from result_contract import build_feature_schema, bind_job, make_result_envelope, decode_result
from post_api_ml import (ReadoutConfig, binary_metrics, prepare_readout_spec,
    fit_correlation_model, predict_correlation_model, evaluate_correlation_model,
    fit_classical_baseline, predict_classical_baseline, evaluate_classical_baseline,
    validate_model, write_model, read_model)

SMILES = ['CC','CCO','C1CC1','CC1CC1','C1CCC1','CC1CCC1',
          'C1CCCC1','CC1CCCC1','C1CCCCC1','CC1CCCCC1','c1ccccc1','Cc1ccccc1',
          'c1ccncc1','Cc1ccncc1','c1cc[nH]c1','Cc1cc[nH]c1',
          'C1CCNCC1','CC1CCNCC1','C1COCCN1','CC1COCCN1']


def synthetic_record(molecule, label, feature_schema, *, source='synthetic_test', heralded=512):
    identity = molecule['molecule_id']
    n = len(feature_schema['selected_features'])
    task = {'schema_version':1,'molecule_id':identity,'endpoint':feature_schema['endpoint'],
        'encoding_fit_id':feature_schema['encoding_fit_id'],'feature_schema_id':feature_schema['feature_schema_id'],
        'bundle_id':sha256_json(['synthetic_bundle',identity]),'inputs_sha256':sha256_json(['synthetic_inputs',identity]),
        'phase_settings_sha256':sha256_json(['synthetic_phases',identity]),
        'deployment_family_id':sha256_json('synthetic_test_family'),
        'lambda_value':1.0,'num_system_qubits':n,'num_index_qubits':1,'shots_requested':8192,
        'source_row_positions':list(molecule['source_row_positions']),
        'source_record_ids':list(molecule['source_record_ids'])}
    task['task_id']=sha256_json(task)
    binding=bind_job(task,'synthetic-job-'+identity)
    # This fixture intentionally encodes the synthetic label in a bit marginal
    # solely to test supervised fitting and serialization, not a quantum claim.
    zero_count=max(1,min(heralded-1,int(heralded*(.8 if label else .2))))
    counts=[{'outputs':[0,0,1.0,0]+[0]*n,'count':zero_count},
            {'outputs':[0,0,1.0,0]+[1]*n,'count':heralded-zero_count},
            {'outputs':[1,0,1.0,0]+[0]*n,'count':8192-heralded}]
    envelope=make_result_envelope(binding,counts,source=source)
    return decode_result(binding,envelope,feature_schema,min_heralded_shots=1)


class PostAPILearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        table=pd.DataFrame({'Drug':SMILES,'Y':[i%2 for i in range(len(SMILES))]})
        clean=clean_endpoint_table(table,'synthetic_endpoint')
        cls.split=split_cleaned_dataset(clean['cleaned'],fractions=(.6,.2,.2))
        cls.encoding=fit_hamiltonian_schema(cls.split['train'],'synthetic_endpoint',CurationConfig(top_k=3,n_bits=64))
        cls.feature_schema=build_feature_schema('synthetic_endpoint',cls.encoding['encoding_fit_id'],
            cls.encoding['active_original_features'],cls.encoding['readout_supports'],
            execution_profile_id=sha256_json('synthetic_declared_physics_profile'))
        cls.transformed={key:transform_molecules(cls.encoding,cls.split[key])['records'] for key in ('train','validation','test')}
        cls.labels={key:dict(zip(cls.split[key]['molecule_id'],cls.split[key]['Y'].astype(int))) for key in ('train','validation','test')}
        cls.records={key:[synthetic_record(row,cls.labels[key][row['molecule_id']],cls.feature_schema)
                         for row in cls.transformed[key]] for key in ('train','validation','test')}
        cls.config=ReadoutConfig(regularization_candidates=(1.,))

    def fit_quantum(self, **overrides):
        options=dict(validation_records=self.records['validation'],validation_labels=self.labels['validation'],
            encoding_schema=self.encoding,feature_schema=self.feature_schema,split_manifest=self.split['manifest'],config=self.config)
        options.update(overrides)
        return fit_correlation_model(self.records['train'],self.labels['train'],**options)

    def test_unfitted_spec_never_invents_quantum_results(self):
        spec=prepare_readout_spec(self.encoding,self.feature_schema,self.split['manifest'])
        self.assertEqual(spec['status'],'awaiting_api_results')
        self.assertIsNone(spec['fitted_coefficients'])
        self.assertIsNone(spec['quantum_prediction_metrics'])
        with self.assertRaisesRegex(ValueError,'missing'):
            fit_correlation_model([],{},validation_records=self.records['validation'],validation_labels=self.labels['validation'],
                encoding_schema=self.encoding,feature_schema=self.feature_schema,split_manifest=self.split['manifest'])

    def test_synthetic_fit_portable_json_roundtrip_and_untouched_test(self):
        model=self.fit_quantum()
        self.assertEqual(model['data_source'],'synthetic_test')
        self.assertFalse(model['test_labels_used_in_fit'])
        self.assertNotIn('test_labels',inspect.signature(fit_correlation_model).parameters)
        result=evaluate_correlation_model(model,self.records['test'],self.labels['test'],feature_schema=self.feature_schema)
        self.assertEqual(result['evaluation_split'],'untouched_test')
        self.assertEqual(result['measurement_source'],'synthetic_test')
        self.assertGreater(result['metrics']['roc_auc'],.99)
        before=predict_correlation_model(model,self.records['test'],feature_schema=self.feature_schema)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'readout.json';write_model(path,model);restored=read_model(path)
            self.assertEqual(model,restored)
            self.assertEqual(before,predict_correlation_model(restored,self.records['test'],feature_schema=self.feature_schema))
        values=np.asarray([r['feature_values'] for r in self.records['train']])
        np.testing.assert_allclose(model['scaler_mean'],values.mean(axis=0))

    def test_label_alignment_uses_identity_not_record_order(self):
        a=self.fit_quantum()
        records=list(reversed(self.records['train']))
        b=fit_correlation_model(records,self.labels['train'],validation_records=self.records['validation'],
            validation_labels=self.labels['validation'],encoding_schema=self.encoding,feature_schema=self.feature_schema,
            split_manifest=self.split['manifest'],config=self.config)
        x=predict_correlation_model(a,self.records['test'],feature_schema=self.feature_schema)
        y=predict_correlation_model(b,self.records['test'],feature_schema=self.feature_schema)
        np.testing.assert_allclose([r['probability_positive'] for r in x],[r['probability_positive'] for r in y],rtol=1e-10)

    def test_wrong_identity_schema_source_and_shot_coverage_rejected(self):
        for mutation in ('missing','duplicate','source','feature','smallshots'):
            rows=copy.deepcopy(self.records['validation'])
            if mutation=='missing':rows=rows[:-1]
            elif mutation=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif mutation=='source':
                row=self.transformed['validation'][0]
                rows[0]=synthetic_record(row,self.labels['validation'][row['molecule_id']],self.feature_schema,source='hardware')
            elif mutation=='feature':rows[0]['feature_values'][0]+=.1
            else:
                row=self.transformed['validation'][0]
                rows[0]=synthetic_record(row,self.labels['validation'][row['molecule_id']],self.feature_schema,heralded=128)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.fit_quantum(validation_records=rows)
        contaminated=copy.deepcopy(self.records['train']);contaminated[0]=copy.deepcopy(self.records['test'][0])
        with self.assertRaises(ValueError):
            fit_correlation_model(contaminated,self.labels['train'],validation_records=self.records['validation'],
                validation_labels=self.labels['validation'],encoding_schema=self.encoding,feature_schema=self.feature_schema,
                split_manifest=self.split['manifest'],config=self.config)

    def test_split_and_encoding_tampering_fails_before_estimator(self):
        split=copy.deepcopy(self.split['manifest']);split['train'][0]=split['test'][0]
        with patch('post_api_ml.LogisticRegression.fit',side_effect=AssertionError('fit must not run')):
            with self.assertRaises(ValueError):self.fit_quantum(split_manifest=split)
        schema=copy.deepcopy(self.encoding);schema['scaling']['feature_max'][0]+=1
        with self.assertRaises(ValueError):self.fit_quantum(encoding_schema=schema)

    def test_valid_readout_schema_cannot_relabel_frozen_fingerprint_wires(self):
        changed=build_feature_schema(self.encoding['endpoint'],self.encoding['encoding_fit_id'],
            [(i+1)%64 for i in self.encoding['active_original_features']],self.encoding['readout_supports'])
        with self.assertRaisesRegex(ValueError,'mapping/supports'):
            prepare_readout_spec(self.encoding,changed,self.split['manifest'])
        records={key:[synthetic_record(row,self.labels[key][row['molecule_id']],changed)
                      for row in self.transformed[key]] for key in ('train','validation')}
        with patch('post_api_ml.LogisticRegression.fit',side_effect=AssertionError('fit must not run')):
            with self.assertRaisesRegex(ValueError,'mapping/supports'):
                fit_correlation_model(records['train'],self.labels['train'],validation_records=records['validation'],
                    validation_labels=self.labels['validation'],encoding_schema=self.encoding,
                    feature_schema=changed,split_manifest=self.split['manifest'],config=self.config)

    def test_changed_training_labels_require_a_new_hamiltonian_fit(self):
        labels=dict(self.labels['train'])
        zero=next(i for i,y in labels.items() if y==0);one=next(i for i,y in labels.items() if y==1)
        labels[zero],labels[one]=1,0  # Class counts stay the same; identities' targets change.
        with patch('post_api_ml.LogisticRegression.fit',side_effect=AssertionError('fit must not run')):
            with self.assertRaisesRegex(ValueError,'Training labels differ'):
                fit_correlation_model(self.records['train'],labels,validation_records=self.records['validation'],
                    validation_labels=self.labels['validation'],encoding_schema=self.encoding,feature_schema=self.feature_schema,
                    split_manifest=self.split['manifest'],config=self.config)
            with self.assertRaisesRegex(ValueError,'Training labels differ'):
                fit_classical_baseline(self.transformed['train'],labels,validation_records=self.transformed['validation'],
                    validation_labels=self.labels['validation'],encoding_schema=self.encoding,split_manifest=self.split['manifest'],config=self.config)

    def test_declared_pair_and_triad_readouts_are_not_dropped_by_the_model(self):
        # An analytic toy interaction family exercises all declared Z orders;
        # it is not reported as a learned real-data Hamiltonian or result.
        encoding=copy.deepcopy(self.encoding)
        encoding['adapter']['hamiltonian_A_interaction_ising'][encoding['endpoint']]['pauli_terms']=[
            {'coefficient':.2,'pauli_string':'Z0 Z1 Z2','locality':3}]
        encoding['hamiltonian_pair_supports']=[];encoding['hamiltonian_triad_supports']=[[0,1,2]]
        encoding['readout_supports']={'singles':[[0],[1],[2]],'pairs':[[0,1],[0,2],[1,2]],'triads':[[0,1,2]]}
        identity=sha256_json({k:v for k,v in encoding.items() if k not in {'schema_id','encoding_fit_id'}})
        encoding['schema_id']=encoding['encoding_fit_id']=identity
        schema=build_feature_schema(encoding['endpoint'],identity,encoding['active_original_features'],encoding['readout_supports'])
        transformed={key:transform_molecules(encoding,self.split[key])['records'] for key in ('train','validation','test')}
        records={key:[synthetic_record(row,self.labels[key][row['molecule_id']],schema) for row in transformed[key]]
                 for key in transformed}
        model=fit_correlation_model(records['train'],self.labels['train'],validation_records=records['validation'],
            validation_labels=self.labels['validation'],encoding_schema=encoding,feature_schema=schema,
            split_manifest=self.split['manifest'],config=self.config)
        self.assertEqual(len(model['feature_names']),7)
        self.assertTrue(any(name.startswith('ZZ[') for name in model['feature_names']))
        self.assertTrue(any(name.startswith('ZZZ[') for name in model['feature_names']))
        self.assertEqual(len(predict_correlation_model(model,records['test'],feature_schema=schema)),len(records['test']))

    def test_test_evaluation_cannot_use_training_or_partial_test(self):
        model=self.fit_quantum()
        with self.assertRaises(ValueError):evaluate_correlation_model(model,self.records['train'],self.labels['train'],feature_schema=self.feature_schema)
        with self.assertRaises(ValueError):evaluate_correlation_model(model,self.records['test'][:-1],self.labels['test'],feature_schema=self.feature_schema)

    def test_validation_selection_changes_no_training_scaler_and_uses_no_test_labels(self):
        model=self.fit_quantum(config=ReadoutConfig(regularization_candidates=(.1,1.,10.)))
        self.assertEqual(len(model['validation_selection']['trials']),3)
        self.assertEqual(set(model['train_molecule_ids']),set(self.split['manifest']['train']))
        self.assertEqual(set(model['test_molecule_ids']),set(self.split['manifest']['test']))
        altered={identity:1-label for identity,label in self.labels['validation'].items()}
        b=self.fit_quantum(validation_labels=altered)
        a=self.fit_quantum()
        self.assertEqual(a['scaler_mean'],b['scaler_mean'])
        self.assertEqual(a['scaler_scale'],b['scaler_scale'])
        self.assertEqual(a['coefficient'],b['coefficient'])

    def test_classical_full_and_selected_count_baselines_match_same_partitions(self):
        for representation,width in (('full_counts',64),('theta',3)):
            model=fit_classical_baseline(self.transformed['train'],self.labels['train'],
                validation_records=self.transformed['validation'],validation_labels=self.labels['validation'],
                encoding_schema=self.encoding,split_manifest=self.split['manifest'],representation=representation,config=self.config)
            self.assertEqual(len(model['coefficient']),width)
            self.assertEqual(set(model['test_molecule_ids']),set(self.split['manifest']['test']))
            report=evaluate_classical_baseline(model,self.transformed['test'],self.labels['test'],encoding_schema=self.encoding)
            self.assertEqual(report['metrics']['samples'],len(self.labels['test']))
            self.assertTrue(0<=report['metrics']['roc_auc']<=1)
            self.assertEqual(len(predict_classical_baseline(model,self.transformed['test'],encoding_schema=self.encoding)),len(self.labels['test']))

    def test_model_tampering_nonfinite_and_unsupported_types_rejected(self):
        model=self.fit_quantum()
        changed=copy.deepcopy(model);changed['coefficient'][0]+=.1
        with self.assertRaisesRegex(ValueError,'checksum'):validate_model(changed)
        changed=copy.deepcopy(model);changed['scaler_scale'][0]=0
        changed['model_id']=sha256_json({k:v for k,v in changed.items() if k!='model_id'})
        with self.assertRaisesRegex(ValueError,'scales'):validate_model(changed)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json';path.write_text('{"model_schema_version":1,"model_schema_version":1}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_model(path)
        labels=copy.deepcopy(self.labels['train']);labels[next(iter(labels))]=1+1j
        with self.assertRaises(ValueError):
            fit_correlation_model(self.records['train'],labels,validation_records=self.records['validation'],
                validation_labels=self.labels['validation'],encoding_schema=self.encoding,feature_schema=self.feature_schema,
                split_manifest=self.split['manifest'],config=self.config)

    def test_configuration_metrics_and_convergence_fail_explicitly(self):
        for config in (ReadoutConfig(regularization_candidates=(True,)),ReadoutConfig(minimum_heralded_shots=0),
                       ReadoutConfig(class_weight='invented'),ReadoutConfig(tolerance=float('nan'))):
            with self.assertRaises(ValueError):config.validate()
        reference=binary_metrics([0,0,1,1],[0,0,1,1])
        self.assertEqual(reference['roc_auc'],1)
        self.assertEqual(reference['brier_score'],0)
        with self.assertRaises(ValueError):binary_metrics([1,1],[.2,.8])
        from sklearn.exceptions import ConvergenceWarning
        with patch('post_api_ml.LogisticRegression.fit',side_effect=ConvergenceWarning('test convergence')):
            with self.assertRaisesRegex(ValueError,'converge'):self.fit_quantum()


if __name__=='__main__':unittest.main()
