from dataclasses import replace
from datetime import datetime
import copy
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np

from data_contract import (CONFIG, FEATURES, Instance, Party, Option, cabin,
                           dump_instance, load_instance, load_private_csvs, validate_config)
from fixtures import small_cases


def processed_tables(case, directory):
    parties = {p.id: p for p in case.parties}
    candidates, features, segments = [], [], []
    for option in case.options:
        p = parties[option.party]
        base = {'RECLOC': p.id, 'OPTION_ID': option.id, 'PAX_CNT': p.pax}
        candidates.append(dict(base))
        features.append({**base, 'CVM': p.cvm, 'DEP_FIRST_DTMZ': option.departure.isoformat(),
                         'ARRIVAL_DELAY_MINS': option.delay_minutes, **dict(zip(FEATURES, option.features))})
        for f, c in option.legs: segments.append({'OPTION_ID': option.id, 'DEP_KEY': f, 'CABIN_USED': c})
    capacities = []
    for f in sorted({f for f, c in case.capacities}):
        capacities.append({'DEP_KEY': f, 'C_AVAIL_CNT': case.capacities.get((f, 'C'), 0),
                           'Y_AVAIL_CNT': case.capacities.get((f, 'Y'), 0)})
    tables = [pd.DataFrame(x) for x in (candidates, segments, features, capacities)]
    names = ['pnr_candidates_clean', 'candidate_segments', 'pnr_option_features_step2', 'available_flights']
    for name, frame in zip(names, tables): frame.to_csv(Path(directory)/(name+'.csv'), index=False)
    return dict(zip(names, tables))


class DataTests(unittest.TestCase):
    def test_synthetic_json_roundtrip_and_no_real_booking_identifiers(self):
        for case in small_cases():
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)/'case.json'; dump_instance(case, p); loaded = load_instance(p)
                self.assertEqual(case.parties, loaded.parties); self.assertEqual(case.options, loaded.options)
                self.assertEqual(dict(case.capacities), dict(loaded.capacities))
                self.assertTrue(all(p.id.startswith('SYN_') for p in loaded.parties))

    def test_missing_and_nonfinite_features_are_not_imputed(self):
        case = small_cases()[0]; original = case.options[0]
        for values in ((0.,), tuple([np.nan]*len(FEATURES)), tuple([np.inf]*len(FEATURES))):
            with self.assertRaises(ValueError): replace(case, options=(replace(original, features=values),))

    def test_invalid_parties_capacities_and_identifiers_fail(self):
        case = small_cases()[0]
        for parties in ((Party('SYN_P', 0, 1.),), (Party('SYN_P', 1.5, 1.),), (Party('SYN_P', 1, np.nan),), (Party(' ', 1, 1.),)):
            with self.assertRaises(ValueError): Instance(parties, (), {})
        for value in (-1, .5, True, np.nan):
            with self.assertRaises(ValueError): replace(case, capacities={('SYN_F', 'Y'): value})
        with self.assertRaises(ValueError): replace(case, parties=(*case.parties, case.parties[0]))
        with self.assertRaises(ValueError): replace(case, options=(*case.options, case.options[0]))

    def test_empty_duplicate_missing_leg_and_naive_time_fail(self):
        case = small_cases()[0]; original = case.options[0]
        for bad in (replace(original, legs=()), replace(original, legs=(('SYN_F', 'Y'), ('SYN_F', 'Y'))),
                    replace(original, legs=(('unknown', 'Y'),)), replace(original, departure=datetime(2025, 1, 1))):
            with self.assertRaises(ValueError): replace(case, options=(bad,))
        for value in ('F', '', 'unknown', None):
            with self.assertRaises(ValueError): cabin(value)
        self.assertEqual(cabin('Business'), 'C'); self.assertEqual(cabin('economy'), 'Y')

    def test_72_hour_cutoff_inclusive_and_no_service_pnr_preserved(self):
        case = small_cases()[3]
        old = next(o for o in case.options if o.party == 'SYN_LATE')
        replacement = replace(old, delay_minutes=4320)
        changed = replace(case, options=tuple(replacement if o.id == old.id else o for o in case.options))
        self.assertIn(replacement, changed.eligible()[0]); self.assertEqual(len(changed.parties), 5)

    def test_processed_adapter_roundtrip_and_schema_consistency(self):
        case = small_cases()[0]
        with tempfile.TemporaryDirectory() as td:
            processed_tables(case, td); loaded, audit = load_private_csvs(td)
            self.assertEqual(loaded.parties, case.parties); self.assertEqual(loaded.options, case.options)
            self.assertEqual(audit['capacity_policy'], 'strict')
            self.assertEqual(audit['cvm_source'], 'CVM')

    def test_adapter_rejects_missing_tables_features_and_metadata_conflicts(self):
        case = small_cases()[0]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError): load_private_csvs(td)
            tables = processed_tables(case, td)
            tables['pnr_option_features_step2'].drop(columns=[FEATURES[0]]).to_csv(Path(td)/'pnr_option_features_step2.csv', index=False)
            with self.assertRaises(ValueError): load_private_csvs(td)
            tables = processed_tables(case, td)
            tables['pnr_candidates_clean'].loc[0, 'PAX_CNT'] = 2
            tables['pnr_candidates_clean'].to_csv(Path(td)/'pnr_candidates_clean.csv', index=False)
            with self.assertRaises(ValueError): load_private_csvs(td)

    def test_capacity_corrections_require_explicit_conservative_policy(self):
        case = small_cases()[0]
        with tempfile.TemporaryDirectory() as td:
            tables = processed_tables(case, td); cap = tables['available_flights']
            cap.loc[0, 'C_AVAIL_CNT'] = -2
            cap = pd.concat([cap, cap.assign(Y_AVAIL_CNT=1)], ignore_index=True)
            cap.to_csv(Path(td)/'available_flights.csv', index=False)
            with self.assertRaises(ValueError): load_private_csvs(td)
            loaded, audit = load_private_csvs(td, capacity_policy='conservative')
            self.assertEqual(loaded.capacities[('SYN_F', 'C')], 0)
            self.assertEqual(loaded.capacities[('SYN_F', 'Y')], 1)
            self.assertEqual(audit['negative_capacities_clipped'], 2)
            self.assertEqual(audit['capacity_conflicts_resolved'], 1)

    def test_departure_times_join_from_candidate_table_when_absent_in_features(self):
        case = small_cases()[0]
        with tempfile.TemporaryDirectory() as td:
            tables = processed_tables(case, td)
            candidates = tables['pnr_candidates_clean'].merge(tables['pnr_option_features_step2'][['OPTION_ID', 'DEP_FIRST_DTMZ']], on='OPTION_ID', validate='one_to_one')
            features = tables['pnr_option_features_step2'].drop(columns=['DEP_FIRST_DTMZ'])
            candidates.to_csv(Path(td)/'pnr_candidates_clean.csv', index=False)
            features.to_csv(Path(td)/'pnr_option_features_step2.csv', index=False)
            loaded, audit = load_private_csvs(td)
            self.assertEqual(loaded.options, case.options)
            self.assertEqual(audit['departure_source'], 'candidates')

    def test_integer_like_inputs_are_canonical_and_cannot_overflow_seat_totals(self):
        from fixtures import option
        from solver import score
        parties=tuple(Party(f'P{j}',np.int8(13),'.5') for j in range(20))
        options=tuple(option(f'O{j}',f'P{j}',[('F','Y')]) for j in range(20))
        instance=Instance(parties,options,{('F','Y'):300})
        self.assertTrue(all(type(p.pax) is int and type(p.cvm) is float for p in instance.parties))
        report=score(instance,{f'P{j}':f'O{j}' for j in range(20)})
        self.assertEqual(report['passengers'],260); self.assertEqual(report['assigned_passengers'],260)
        self.assertEqual(report['capacity_violations'],0)
        numeric_strings=replace(instance,parties=(Party('P0','13','.5'),),options=(replace(options[0],features=tuple(str(x) for x in options[0].features)),))
        self.assertTrue(all(type(x) is float for x in numeric_strings.options[0].features))
        self.assertEqual(score(numeric_strings,{'P0':'O0'})['assigned_passengers'],13)

    def test_bad_config_is_rejected(self):
        for key, value in [('unassigned_cost', np.nan), ('block_hours', 0), ('seeds', [7, 7])]:
            bad = copy.deepcopy(CONFIG); bad[key] = value
            with self.assertRaises(ValueError): validate_config(bad)
