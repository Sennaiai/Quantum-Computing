from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from data_contract import Instance, Party, FEATURES, instance_fingerprint, dump_instance, load_instance
from connections import (SCALES, validate_network, validate_policy, option_from_route, expand_connections,
                         route_violations, route_quality, bind_and_verify_features)
from connection_curation import build_connection_metadata

POLICY = {'min_connection_minutes':45,'max_layover_minutes':1080,'recovery_minutes':4320,'max_legs':3,'path_budget_per_journey':20000}


def network_case():
    rows = [dict(id='D',origin='A',destination='B',departure=0,arrival=3600,total_availability=3),
        dict(id='F1',origin='A',destination='H',departure=0,arrival=1800,total_availability=3),
        dict(id='F2',origin='H',destination='B',departure=4500,arrival=6300,total_availability=3),
        dict(id='CYCLE',origin='H',destination='A',departure=4500,arrival=6300,total_availability=3)]
    caps = {(r['id'],c):n for r in rows for c,n in [('C',1),('Y',2)]}
    p = Party('P',1,.5); empty = Instance((p,),(),caps)
    raw = {'schema_version':1,'base_catalog_sha256':'abc','flights':rows,
        'journeys':{'P':dict(origin='A',destination='B',departure=0,arrival=3600,duration=60,connections=0,cabin='Y')},
        'feature_ranges':{'ARRIVAL_DELAY_MINS':[-4320,4320],'EXCESS_TRAVEL_TIME_MINS':[0,4320],
            'CONN_PENALTY':[0,2],'CAPACITY_STRESS_PROXY':[0,2],'BOTTLENECK_RATIO':[0,1]}}
    network = validate_network(raw,empty,'abc'); network['priorities']={'P':.5}
    option = option_from_route(empty,network,'P',(('D','Y'),),'BASE')
    return Instance((p,),(option,),caps), raw


class ConnectionGenerationTests(unittest.TestCase):
    def expand(self, instance, raw, policy=None):
        return expand_connections(instance,validate_network(raw,instance,'abc'), POLICY if policy is None else policy)

    def test_both_cabins_and_connecting_routes_are_included_without_cycles(self):
        i,m = network_case(); expanded,network,audit = self.expand(i,m)
        self.assertEqual(len(expanded.options),6)
        self.assertEqual(audit['added_options'],5)
        self.assertTrue(any(len(o.legs)==2 for o in expanded.options))
        self.assertTrue(any(o.legs==(('D','C'),) for o in expanded.options))
        self.assertTrue(all(not route_violations(expanded,network,o.party,o.legs,POLICY) for o in expanded.options))
        self.assertFalse(any(f=='CYCLE' for o in expanded.options for f,c in o.legs))
        self.assertEqual(i.options[0], expanded.options[0])

    def test_exact_minimum_transfer_and_maximum_layover_boundaries(self):
        for wait, expected in [(45*60,6),(45*60-1,2),(1080*60,6),(1080*60+1,2)]:
            i,m = network_case(); m['flights'][2]['departure']=1800+wait; m['flights'][2]['arrival']=1800+wait+1800
            with self.subTest(wait=wait): self.assertEqual(len(self.expand(i,m)[0].options),expected)

    def test_final_arrival_deadline_is_enforced_not_just_last_departure(self):
        for arrival, expected in [(72*3600,6),(72*3600+1,2)]:
            i,m = network_case(); m['flights'][2]['arrival']=arrival
            with self.subTest(arrival=arrival): self.assertEqual(len(self.expand(i,m)[0].options),expected)

    def test_party_capacity_eliminates_an_infeasible_cabin(self):
        i,m = network_case(); caps=dict(i.capacities); caps[('F2','C')]=0; i=replace(i,capacities=caps)
        e,_,_=self.expand(i,m)
        self.assertFalse(any(('F2','C') in o.legs for o in e.options))
        self.assertTrue(any(('F2','Y') in o.legs for o in e.options))

    def test_path_budget_fails_explicitly_instead_of_truncating(self):
        i,m=network_case()
        with self.assertRaisesRegex(ValueError,'no silent route truncation'):
            self.expand(i,m,{**POLICY,'path_budget_per_journey':1})

    def test_network_binding_rejects_wrong_catalog_or_incomplete_inventory(self):
        i,m=network_case()
        with self.assertRaises(ValueError): validate_network(m,i,'different')
        with self.assertRaises(ValueError): validate_network({**m,'flights':m['flights'][:-1]},i,'abc')
        wrong=deepcopy(m); wrong['flights'].append(wrong['flights'][0])
        with self.assertRaises(ValueError): validate_network(wrong,i,'abc')
        wrong=deepcopy(m); wrong['flights'][0]['arrival']=-1
        with self.assertRaises(ValueError): validate_network(wrong,i,'abc')

    def test_original_feature_drift_is_rejected_and_private_raw_cvm_is_not_substituted(self):
        i,m=network_case(); feature=list(i.options[0].features); feature[0]+=.01
        wrong=replace(i,options=(replace(i.options[0],features=tuple(feature)),))
        with self.assertRaisesRegex(ValueError,'reproduce original'): self.expand(wrong,m)
        raw_cvm=replace(i,parties=(replace(i.parties[0],cvm=10000),))
        e,_,audit=self.expand(raw_cvm,m)
        self.assertEqual(audit['base_feature_maximum_absolute_error'],0)
        self.assertTrue(all(o.features[FEATURES.index('CVM_NORM')]==.5 for o in e.options))

    def test_service_proxies_and_empty_solution_do_not_claim_satisfaction_measurements(self):
        i,m=network_case(); e,n,_=self.expand(i,m)
        connecting=next(o for o in e.options if len(o.legs)==2 and all(c=='Y' for f,c in o.legs))
        q=route_quality(e,{'P':connecting.id},n)
        self.assertEqual(q['mean_connections'],1); self.assertEqual(q['mean_journey_minutes'],105)
        self.assertEqual(q['connecting_pnrs'],1); self.assertEqual(q['downgraded_pnrs'],0)
        self.assertIsNone(route_quality(e,{},n)['mean_connections'])
        self.assertIn('no measured customer',q['interpretation'])

    def test_policy_invalid_waits_and_counts_are_rejected(self):
        for changes in [{'max_legs':4},{'min_connection_minutes':1081},{'path_budget_per_journey':True}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): validate_policy({**POLICY,**changes})

    def test_canonical_fingerprint_survives_serialization(self):
        i,_=network_case()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'catalog.json.gz'; dump_instance(i,p)
            self.assertEqual(instance_fingerprint(i),instance_fingerprint(load_instance(p)))


class ConnectionCurationTests(unittest.TestCase):
    def tables(self, folder):
        pd.DataFrame([dict(RECLOC='PRIVATE_BOOKING',OPTION_ID='PRIVATE_OPTION',DEP_FIRST_DTMZ='2020-01-01T00:00:00Z',ARR_FINAL_DTMZ='2020-01-01T01:00:00Z',CABIN_ORIG='Y')]).to_csv(folder/'pnr_candidates_clean.csv',index=False)
        pd.DataFrame([dict(RECLOC='PRIVATE_BOOKING',OPTION_ID='PRIVATE_OPTION',DEPARTURE_DELAY_MINS=0,ARRIVAL_DELAY_MINS=0,ORIG_TRAVEL_TIME_MINS=60,ORIG_NUM_CONNS=0,EXCESS_TRAVEL_TIME_MINS=0,CONN_PENALTY=0,CAPACITY_STRESS_PROXY=.25,BOTTLENECK_RATIO=.5)]).to_csv(folder/'pnr_option_features_step2.csv',index=False)
        pd.DataFrame([dict(OPTION_ID='PRIVATE_OPTION',SEG_SEQ=1,ORIG_CD='PRIVATE_ORIGIN',DEST_CD='PRIVATE_DEST')]).to_csv(folder/'candidate_segments.csv',index=False)
        pd.DataFrame([dict(DEP_KEY='PRIVATE_FLIGHT',ORIG_CD='PRIVATE_ORIGIN',DEST_CD='PRIVATE_DEST',DEP_DTMZ='2020-01-01T00:00:00Z',ARR_DTMZ='2020-01-01T01:00:00Z',TOTAL_AVAIL=4)]).to_csv(folder/'available_flights.csv',index=False)
        return {'pnrs':{'PRIVATE_BOOKING':'P'},'flights':{'PRIVATE_FLIGHT':'F'},'time_shift_seconds':86400}

    def test_metadata_removes_booking_flight_option_and_airport_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); mapping=self.tables(folder); data=build_connection_metadata(folder,mapping,'abc')
        text=json.dumps(data)
        for value in ('PRIVATE_BOOKING','PRIVATE_OPTION','PRIVATE_FLIGHT','PRIVATE_ORIGIN','PRIVATE_DEST'):
            self.assertNotIn(value,text)
        self.assertEqual(set(data['journeys']),{'P'})
        self.assertEqual(data['flights'][0]['departure'],1577836800+86400)

    def test_missing_journey_metadata_is_rejected_without_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); mapping=self.tables(folder); path=folder/'pnr_option_features_step2.csv'
            frame=pd.read_csv(path).drop(columns='ORIG_NUM_CONNS'); frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Incomplete connection source'):
                build_connection_metadata(folder,mapping,'abc')
