"""Original synthetic solver fixtures; no airline records or identifiers."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from data_contract import Party, Option, Instance, FEATURES, dump_instance

EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)


def option(oid, party, legs, *, hour=8, cvm=.5, delay=60, cost_level=.1, mixed=0):
    features = {'ARRIVAL_DELAY_MINS_NORM': cost_level,
                'EXCESS_TRAVEL_TIME_MINS_NORM': cost_level,
                'CONN_PENALTY_NORM': cost_level,
                'CAPACITY_STRESS_PROXY_NORM': cost_level,
                'BOTTLENECK_RATIO_NORM': cost_level,
                'CVM_NORM': cvm, 'MIXED_CABIN_FLAG': mixed, 'CABIN_DISP': 0.}
    return Option(oid, party, tuple(legs), EPOCH+timedelta(hours=hour), delay,
                  tuple(float(features[k]) for k in FEATURES))


def small_cases():
    triple = Instance(tuple(Party(f'SYN_P{i}', 1, i/3) for i in range(1, 4)),
        tuple(option(f'SYN_O{i}', f'SYN_P{i}', [('SYN_F', 'Y')], cvm=i/3) for i in range(1, 4)),
        {('SYN_F', 'Y'): 2}, 'collective_capacity')
    cross = Instance((Party('SYN_A', 1, .2), Party('SYN_B', 1, .9)),
        (option('SYN_A1', 'SYN_A', [('SYN_IN_A', 'Y'), ('SYN_SHARED', 'Y')], hour=8, cvm=.2),
         option('SYN_B1', 'SYN_B', [('SYN_IN_B', 'Y'), ('SYN_SHARED', 'Y')], hour=13, cvm=.9)),
        {('SYN_IN_A', 'Y'): 1, ('SYN_IN_B', 'Y'): 1, ('SYN_SHARED', 'Y'): 1}, 'cross_block_capacity')
    trap = Instance((Party('SYN_A', 1, .9), Party('SYN_B', 1, .2)),
        (option('SYN_A1', 'SYN_A', [('SYN_F1', 'Y')], cvm=.9, cost_level=.05),
         option('SYN_A2', 'SYN_A', [('SYN_F2', 'Y')], hour=15, cvm=.9, cost_level=.1),
         option('SYN_B1', 'SYN_B', [('SYN_F1', 'Y')], cvm=.2, cost_level=.05)),
        {('SYN_F1', 'Y'): 1, ('SYN_F2', 'Y'): 1}, 'greedy_alternative_trap')
    eligibility = Instance((Party('SYN_A', 2, .8), Party('SYN_B', 1, .6), Party('SYN_NONE', 1, .3),
                            Party('SYN_LATE', 1, .4), Party('SYN_BIG', 3, .5)),
        (option('SYN_A1', 'SYN_A', [('SYN_F1', 'C'), ('SYN_F2', 'Y')], cvm=.8, mixed=1),
         option('SYN_B1', 'SYN_B', [('SYN_F1', 'Y')], cvm=.6),
         option('SYN_LATE1', 'SYN_LATE', [('SYN_F1', 'Y')], cvm=.4, delay=4321),
         option('SYN_BIG1', 'SYN_BIG', [('SYN_F2', 'Y')], cvm=.5)),
        {('SYN_F1', 'C'): 2, ('SYN_F1', 'Y'): 1, ('SYN_F2', 'Y'): 2}, 'eligibility_and_cabins')
    return (triple, cross, trap, eligibility)


def scale_case(parties=72, seed=803):
    rng = np.random.default_rng(seed)
    caps = {(f'SYN_F{i:02}', cab): int(rng.integers(2, 4) if cab == 'C' else rng.integers(5, 9))
            for i in range(12) for cab in ('C', 'Y')}
    groups, options = [], []
    for i in range(parties):
        pid = f'SYN_P{i:04}'; pax = int(rng.integers(1, 4)); cvm = float(rng.uniform(.05, 1.))
        groups.append(Party(pid, pax, cvm))
        for j in range(3):
            indices = rng.choice(12, size=int(rng.integers(1, 3)), replace=False)
            legs = [(f'SYN_F{k:02}', 'C' if rng.random() < .15 else 'Y') for k in indices]
            options.append(option(f'SYN_O{i:04}_{j}', pid, legs,
                           hour=int(rng.integers(0, 48)), cvm=cvm,
                           delay=int(rng.integers(0, 4200)), cost_level=float(rng.uniform(.02, .8))))
    return Instance(tuple(groups), tuple(options), caps, f'synthetic_{parties}_pnrs')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent/'fixtures'; root.mkdir(exist_ok=True)
    for instance in small_cases(): dump_instance(instance, root/(instance.name+'.json'))
    print('Wrote four synthetic adversarial input fixtures')
