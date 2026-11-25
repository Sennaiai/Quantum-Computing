from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from data_contract import dump_instance, load_instance
from fixtures import small_cases
from deidentify import deidentify
from solver import solve_decomposed, score


class DeidentificationTests(unittest.TestCase):
    def test_problem_and_deterministic_solver_are_preserved(self):
        for source in small_cases()[:3]:
            public, mapping, audit = deidentify(source)
            self.assertFalse(audit['synthetic'])
            self.assertEqual(sum(p.pax for p in source.parties), sum(p.pax for p in public.parties))
            a, ar = solve_decomposed(source, reads=12, sweeps=100)
            b, br = solve_decomposed(public, reads=12, sweeps=100)
            self.assertEqual({mapping['pnrs'][p]: mapping['options'][o] for p, o in a.items()}, b)
            for field in ('objective', 'pnr_allocation_percent', 'passenger_allocation_percent', 'capacity_violations'):
                self.assertAlmostEqual(ar[field], br[field], places=9)
            self.assertTrue(all(p.id.startswith('PNR_') for p in public.parties))
            self.assertTrue(all(o.id.startswith('OPT_') for o in public.options))
            self.assertTrue(all(f.startswith('LEG_') for f, c in public.capacities))

    def test_compressed_catalog_roundtrip(self):
        source = small_cases()[0]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'catalog.json.gz'; dump_instance(source, path)
            restored = load_instance(path)
            self.assertEqual(restored.parties, source.parties)
            self.assertEqual(restored.options, source.options)

    def test_changed_priority_order_is_not_silently_published(self):
        source = small_cases()[0]
        groups = tuple(replace(p, cvm=-p.cvm+2) for p in source.parties)
        with self.assertRaises(ValueError): deidentify(replace(source, parties=groups))
