"""Fixed-shape named API entrypoints, with native execution still unverified.

The local numeric protocol is a mathematical transport contract. The published
Triple Alpha SDK instead accepts named inputs. One generated entrypoint per
deployment family gives every scalar and measurement a unique exported name;
it makes no assumption about queues for repeated input variables in loops.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from artifact_paths import validate_output_dir
from runtime_input_protocol import (atomic_write_json, canonical_json,
                                    validate_runtime_input_document)
from runtime_sync_validation import QBE_MODULES

SDK_VERSION = "0.2.3"
SDK_WHEEL_SHA256 = "6c51598a53d4cdc3b0402d82665467251e4975ca269d4f92af8f95ac0127f75d"


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def input_slots(family):
    nq, nt = family["num_system_qubits"], family["num_terms"]
    values = [f"theta_{i}" for i in range(nq)]
    for term in range(nt):
        values.append(f"coefficient_{term}")
        values.extend(f"pauli_{term}_{wire}" for wire in range(nq))
    values.extend(f"cos_phase_{i}" for i in range(family["num_cosine_phases"]))
    values.extend(f"sin_phase_{i}" for i in range(family["num_sine_phases"]))
    values.append("branch_angle")
    return values


def output_names(family):
    return (["result_branch", "result_real", "lambda"]
            + [f"result_index_{i}" for i in range(family["num_index_qubits"])]
            + [f"result_system_{i}" for i in range(family["num_system_qubits"])])


def render_native_main(family):
    """Generate candidate Beryllium using only already-used source constructs."""
    for key in ("num_system_qubits", "num_index_qubits", "num_terms", "num_cosine_phases", "num_sine_phases"):
        if type(family.get(key)) is not int or not 1 <= family[key] <= 10000:
            raise ValueError("Native family dimensions must be bounded positive integers")
    nq, ni, nt, nc, ns = (family[k] for k in ("num_system_qubits", "num_index_qubits", "num_terms", "num_cosine_phases", "num_sine_phases"))
    if len(input_slots(family)) > 100000 or ni != max(1, (nt-1).bit_length()) or nc % 2 != 1 or ns % 2 != 0:
        raise ValueError("Native family dimensions/parity exceed the local contract")
    lines = ['# Generated fixed-shape candidate. Native compilation/gates remain unverified.',
             'import "math.qbe" as math', 'import "qsvt_mod.qbe" as qsvt_mod', 'func main(){']
    lines += [f'    input -> num {name}' for name in input_slots(family)]
    lines += [f'    {nt} -> num num_coefs', f'    {nq} -> num num_qubits', f'    {ni} -> num num_ancillas',
              '    num[] coefs', '    num[] sign', '    num[] pauli_strings', '    num[] cos_phases', '    num[] sin_phases',
              '    qusys(2)[] qubits', '    qusys(2)[] ancillas']
    for i in range(nq):
        lines += [f'    qubit -> qubits.[{i+1}]', f'    ry(3.141592653589793 * theta_{i})[qubits.[{i+1}]]']
    for i in range(ni):
        lines.append(f'    qubit -> ancillas.[{i+1}]')
    for i in range(nt):
        lines.append(f'    coefficient_{i} -> coefs.[{i+1}]')
        for q in range(nq):
            lines.append(f'    pauli_{i}_{q} -> pauli_strings.[{i*nq+q+1}]')
    for prefix, count, array in (("cos_phase", nc, "cos_phases"), ("sin_phase", ns, "sin_phases")):
        lines += [f'    {prefix}_{i} -> {array}.[{i+1}]' for i in range(count)]
    lines += ['    0 -> num lambda',
        '    for {1 -> num i}{i <= num_coefs}{i + 1 -> i}{',
        '        if {coefs.[i] < 0}{', '            1 -> sign.[i]', '            -1 * coefs.[i] -> coefs.[i]',
        '        } else {', '            0 -> sign.[i]', '        }',
        '        lambda + coefs.[i] -> lambda', '    }',
        '    for {1 -> num i}{i <= num_coefs}{i + 1 -> i}{',
        '        coefs.[i] / lambda -> num normalized_weight',
        '        math.sqrt(normalized_weight) -> num amplitude_value', '        amplitude_value -> coefs.[i]', '    }',
        '    qubit -> branch', '    qubit -> real_ancilla',
        '    QSVT(num_coefs, num_ancillas, num_qubits) -> QSVT qsvt',
        f'    qsvt.apply(ancillas, qubits, branch, real_ancilla, coefs, pauli_strings, sign, cos_phases, {nc}, sin_phases, {ns}, branch_angle)',
        '    measure[branch] -> num result_branch', '    measure[real_ancilla] -> num result_real',
        '    result_branch -> output', '    result_real -> output', '    lambda -> output']
    for label, array, count in (("index", "ancillas", ni), ("system", "qubits", nq)):
        for i in range(count):
            lines += [f'    measure[{array}.[{i+1}]] -> num result_{label}_{i}', f'    result_{label}_{i} -> output']
    return "\n".join(lines+['}', ''])


def write_native_family(family, source_dir, output_dir):
    """Publish a source/binding bundle; compilation is deliberately not attested."""
    output = validate_output_dir(output_dir)
    main = render_native_main(family)
    names = input_slots(family)
    if re.findall(r"input\s*->\s*num\s+(\w+)", main) != names or len(set(names)) != len(names):
        raise ValueError("Generated named input sites are inconsistent")
    sources = {name: (Path(source_dir)/name).read_bytes() for name in QBE_MODULES if name != "main.qbe"}
    sources["main.qbe"] = main.encode()
    binding = {"schema_version": 1, "deployment_family_id": family["deployment_family_id"],
        "shape": {k:family[k] for k in ("num_system_qubits", "num_index_qubits", "num_terms", "num_cosine_phases", "num_sine_phases")},
        "input_names": names, "output_names": output_names(family),
        "input_wire_type": "finite numeric strings in uniquely named [name,value] pairs",
        "sdk_version": SDK_VERSION, "sdk_wheel_sha256": SDK_WHEEL_SHA256,
        "module_hashes": {name: hashlib.sha256(value).hexdigest() for name,value in sorted(sources.items())},
        "native_compiler_verified": False, "gate_convention_verified": False,
        "native_input_binding_verified": False, "native_output_mapping_verified": False,
        "scope": "generated source read/write sites and SDK request shape only"}
    binding["binding_id"] = _hash(binding)
    directory = output/binding["binding_id"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in sources.items():
        path = directory/name
        if path.exists() and path.read_bytes() != value:
            raise ValueError("Existing native source bundle differs from its identity")
        path.write_bytes(value)
    atomic_write_json(directory/"binding.json", binding)
    return directory, binding


def named_inputs(document, binding):
    """Map a verified logical stream to a fixed source-level named entrypoint."""
    decoded = validate_runtime_input_document(document)
    if not isinstance(binding, dict) or binding.get("binding_id") != _hash({k:v for k,v in binding.items() if k != "binding_id"}):
        raise ValueError("Native binding identity differs from its contents")
    shape = binding["shape"]
    for key, actual in (("num_terms", decoded["num_terms"]), ("num_system_qubits", decoded["num_system_qubits"]),
                        ("num_index_qubits", decoded["num_index_qubits"]), ("num_cosine_phases", decoded["num_cosine_phases"]),
                        ("num_sine_phases", decoded["num_sine_phases"])):
        if shape[key] != actual:
            raise ValueError("Runtime dimensions differ from generated deployment family")
    names = input_slots(shape)
    if binding["input_names"] != names or binding["output_names"] != output_names(shape):
        raise ValueError("Native input/output name map differs from the fixed shape")
    values = document["inputs"][5:]
    if len(names) != len(values):
        raise ValueError("Named input count differs from the numeric protocol")
    return [[name, format(float(value), ".17g")] for name,value in zip(names, values)]
