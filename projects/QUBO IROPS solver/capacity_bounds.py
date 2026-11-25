"""Verify exact rational upper bounds on allocation, independently of any solver.

For every option, y_PNR + pax*sum(y_leg) >= its allocation benefit,
with all y >= 0. Multiplying by feasible option choices proves that
sum(y_PNR)+sum(capacity*y_leg) bounds every feasible allocation.
The certificate is checked with Fraction arithmetic, not LP tolerances.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from data_contract import ROOT, load_instance


def verify_capacity_bounds(instance, certificate, catalog_sha256):
    if certificate.get('catalog_sha256') != catalog_sha256:
        raise ValueError('Capacity certificate refers to a different catalog')
    if set(certificate['bounds']) != {'pnrs', 'passengers'}:
        raise ValueError('Certificate must bind both allocation measures')
    parties = instance.party_by_id; options = instance.eligible()[0]; out = {}
    for target, data in certificate['bounds'].items():
        pdual = {p: Fraction(v) for p, v in data['pnr_dual'].items()}
        ldual = {}
        for f, c, value in data['leg_dual']:
            key = (f, c)
            if key in ldual: raise ValueError('Duplicate dual resource')
            ldual[key] = Fraction(value)
        if not set(pdual) <= set(parties) or not set(ldual) <= set(instance.capacities):
            raise ValueError('Certificate contains unknown labels')
        if any(v < 0 for v in (*pdual.values(), *ldual.values())):
            raise ValueError('Dual weights must be nonnegative')
        for option in options:
            benefit = 1 if target == 'pnrs' else parties[option.party].pax
            lhs = pdual.get(option.party, Fraction(0))+parties[option.party].pax*sum(
                (ldual.get(k, Fraction(0)) for k in option.legs), Fraction(0))
            if lhs < benefit: raise ValueError('Capacity certificate fails an option inequality')
        bound = sum(pdual.values(), Fraction(0))+sum(
            (instance.capacities[k]*v for k, v in ldual.items()), Fraction(0))
        denominator = len(parties) if target == 'pnrs' else sum(p.pax for p in instance.parties)
        if str(bound) != data['rational_upper_bound'] or data['denominator'] != denominator:
            raise ValueError('Capacity certificate bound/denominator mismatch')
        integer_bound = bound.numerator//bound.denominator
        if data['integer_upper_bound'] != integer_bound: raise ValueError('Incorrect integer upper bound')
        out[target] = {'rational_upper_bound': str(bound), 'integer_upper_bound': integer_bound,
                       'denominator': denominator, 'fractional_upper_bound_percent': 100*float(bound)/denominator,
                       'integer_upper_bound_percent': 100*integer_bound/denominator,
                       'seventy_percent_possible': integer_bound*100 >= denominator*70}
    return out


def verify_public_bounds(instance=None):
    path = ROOT/'data/irops_catalog.json.gz'
    if instance is None: instance = load_instance(path)
    certificate = json.loads((ROOT/'data/capacity_bounds.json').read_text())
    return verify_capacity_bounds(instance, certificate, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__': print(json.dumps(verify_public_bounds(), indent=2))
