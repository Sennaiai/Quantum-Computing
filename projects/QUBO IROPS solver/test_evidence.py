from pathlib import Path
from collections import Counter
from copy import deepcopy
import hashlib
import json
import unittest
import nbformat
import pandas as pd
from experiment import ROOT, NOTEBOOK, CONFIG, load_experiment_catalogs, source_hashes
from data_contract import instance_fingerprint
from capacity_bounds import verify_capacity_bounds
from solver import score


class ActualDataEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control, cls.instance, cls.network, cls.provenance, cls.audit = load_experiment_catalogs()

    def test_actual_catalog_and_deidentification_contract(self):
        i,p=self.instance,self.provenance
        self.assertFalse(p['synthetic'])
        self.assertEqual((len(i.parties),len(i.options),sum(p.pax for p in i.parties)),(7653,439145,11822))
        self.assertEqual(len(self.control.options),57830)
        self.assertTrue(all(p.id.startswith('PNR_') for p in i.parties))
        self.assertTrue(all(o.id.startswith(('OPT_','EXT_')) for o in i.options))
        self.assertTrue(all(f.startswith('LEG_') for f,c in i.capacities))
        self.assertTrue(all(row[key].startswith('APT_') for row in self.network['flights'].values() for key in ('origin','destination')))
        self.assertEqual(p['input_audit']['negative_capacities_clipped'],622)
        self.assertEqual(self.audit['removed_reasons'],{'late_recovery':133})
        self.assertEqual(self.audit['added_options'],381315)
        self.assertLess(self.audit['base_feature_maximum_absolute_error'],1e-12)

    def test_all_expanded_routes_have_independently_validated_times_endpoints_cabins_and_capacity(self):
        flights=self.network['flights']; journeys=self.network['journeys']; parties=self.instance.party_by_id
        for option in self.instance.options:
            path=[flights[f] for f,c in option.legs]; journey=journeys[option.party]
            self.assertLessEqual(len(path),3)
            self.assertEqual(path[0]['origin'],journey['origin']); self.assertEqual(path[-1]['destination'],journey['destination'])
            self.assertEqual(len({path[0]['origin'],*[r['destination'] for r in path]}),len(path)+1)
            self.assertGreaterEqual(path[0]['departure'],journey['departure'])
            self.assertLessEqual(path[-1]['arrival'],journey['departure']+72*3600)
            for a,b in zip(path,path[1:]):
                self.assertEqual(a['destination'],b['origin'])
                self.assertGreaterEqual(b['departure']-a['arrival'],45*60)
                self.assertLessEqual(b['departure']-a['arrival'],18*3600)
            for leg in option.legs: self.assertLessEqual(parties[option.party].pax,self.instance.capacities[leg])

    def test_published_manifest_and_source_identities(self):
        manifest=json.loads((ROOT/'results/manifest.json').read_text())
        self.assertEqual(set(manifest['files']),{'actual_catalog_results.json','assignments.csv.gz','allocation_resources.svg'})
        for name,value in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest(),value)
        result=json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        self.assertEqual(result['source_hashes'],source_hashes()); self.assertEqual(result['status'],'passed')
        self.assertEqual(result['expanded_instance_fingerprint'],instance_fingerprint(self.instance))

    def test_every_seed_and_control_allocation_capacity_and_route_count_recomputed(self):
        result=json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        df=pd.read_csv(ROOT/'results/assignments.csv.gz',dtype='string')
        self.assertEqual(list(df.PNR),[p.id for p in self.instance.parties]); self.assertFalse(df.PNR.duplicated().any())
        expected={'PNR',*[f'{prefix}seed_{seed}' for seed in CONFIG['seeds'] for prefix in ('','control_')]}
        self.assertEqual(set(df),expected)
        for rows,instance,prefix in [(result['runs'],self.instance,''),(result['controls'],self.control,'control_')]:
            self.assertEqual([r['seed'] for r in rows],CONFIG['seeds'])
            for row in rows:
                selected={str(p):str(o) for p,o in zip(df.PNR,df[f"{prefix}seed_{row['seed']}"]) if not pd.isna(o)}
                actual=score(instance,selected,result['scenario'])
                for field in actual: self.assertAlmostEqual(actual[field],row[field],places=7)
                # Independent integer accounting, separate from solver.loads_for.
                counts=Counter(); passengers=0; connections=0
                for pid,oid in selected.items():
                    option=instance.option_by_id[oid]; self.assertEqual(option.party,pid)
                    pax=instance.party_by_id[pid].pax; passengers+=pax; connections+=len(option.legs)-1
                    for leg in option.legs: counts[leg]+=pax
                self.assertTrue(all(load<=instance.capacities[leg] for leg,load in counts.items()))
                self.assertEqual((len(selected),passengers),(row['assigned_pnrs'],row['assigned_passengers']))
                self.assertAlmostEqual(connections/len(selected),row['service_proxies']['mean_connections'])
                self.assertEqual(actual['capacity_violations'],0)
                self.assertLessEqual(row['resources']['peak_variables'],CONFIG['max_bqm_variables'])
                for chunk in row['chunks']:
                    self.assertGreater(chunk['penalty'],chunk['objective_range_bound'])
                    self.assertLessEqual(chunk['best_sample_energy_identity_error'],chunk['energy_identity_roundoff_bound'])
                before=row['allocation_refinement']['before']
                self.assertGreaterEqual(row['assigned_pnrs'],before['assigned_pnrs'])
                self.assertGreaterEqual(row['assigned_passengers'],before['assigned_passengers'])
                self.assertLessEqual(row['objective'],before['objective']+1e-7)
        for row,control in zip(result['runs'],result['controls']):
            self.assertGreaterEqual(row['assigned_pnrs'],control['assigned_pnrs'])
            self.assertGreaterEqual(row['assigned_passengers'],control['assigned_passengers'])
            self.assertLessEqual(row['objective'],control['objective']+1e-7)

    def test_expanded_capacity_certificates_bind_the_actual_generated_instance(self):
        cert=json.loads((ROOT/'data/expanded_capacity_bounds.json').read_text()); fingerprint=instance_fingerprint(self.instance)
        actual=verify_capacity_bounds(self.instance,cert,fingerprint)
        report=json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        self.assertEqual(report['capacity_upper_bounds'],actual)
        with self.assertRaises(ValueError): verify_capacity_bounds(self.instance,cert,'different-instance')
        wrong=deepcopy(cert); wrong['bounds']['pnrs']['pnr_dual']={}; wrong['bounds']['pnrs']['leg_dual']=[]
        with self.assertRaises(ValueError): verify_capacity_bounds(self.instance,wrong,fingerprint)

    def test_notebook_is_concise_and_does_not_load_private_source_tables(self):
        self.assertEqual({p.name for p in ROOT.glob('*.ipynb')},{NOTEBOOK})
        nb=nbformat.read(ROOT/NOTEBOOK,as_version=4); nbformat.validate(nb)
        source='\n'.join(c.source for c in nb.cells)
        self.assertNotIn('!pip',source); self.assertNotIn('irops_clean',source)
        self.assertLess(sum(len(c.source.split()) for c in nb.cells if c.cell_type=='markdown'),180)
        self.assertFalse(any(o.output_type=='error' for c in nb.cells for o in c.get('outputs',[])))

    def test_external_client_comparison_is_not_fabricated(self):
        result=json.loads((ROOT/'results/actual_catalog_results.json').read_text())
        self.assertFalse(result['client_comparison']['outperformance_claim'])
        self.assertIn('independently',result['client_comparison']['scope'])
        self.assertIn('no quantum advantage',result['comparison'])

    def test_end_to_end_runtime_covers_both_variants_and_all_solver_runs(self):
        result=json.loads((ROOT/'results/actual_catalog_results.json').read_text()); timing=result['timing']
        self.assertEqual(timing['seed_count'],len(CONFIG['seeds'])); self.assertEqual(timing['variants_per_seed'],2)
        self.assertGreaterEqual(timing['end_to_end_seconds'],sum(r['seconds'] for rows in (result['runs'],result['controls']) for r in rows))
        self.assertIn('exports',timing['scope'])
        for rows in (result['runs'],result['controls']):
            for r in rows: self.assertGreaterEqual(r['seconds'],r['decomposed_sa_seconds']+r['allocation_refinement']['seconds'])
