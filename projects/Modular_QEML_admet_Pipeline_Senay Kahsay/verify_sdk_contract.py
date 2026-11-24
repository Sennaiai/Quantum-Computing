"""Check the pinned SDK with mocked transports only; never call a deployment.

Run in the isolated requirements-sdk.txt environment. This establishes SDK
request/outer-response compatibility, not native compilation or service output
mapping. SDK source: https://pypi.org/project/triple-alpha/0.2.3/ .
"""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"python_classes")]


def main():
    def denied(*args, **kwargs):
        raise RuntimeError("SDK contract verification forbids network access")
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    assert version("triple-alpha") == "0.2.3"
    from triplealpha import SynchronousRunner
    from triplealpha.models.api.execution import (ExecutionCreateRequest, ExecutionInputsUpdateRequest,
        ExecutionResultsResponse, ExecutionStatusResponse)
    from triplealpha.utils.constants import JobStatus
    from triple_alpha_client import TripleAlphaClient
    values = [["theta_0", "0.5"], ["coefficient_0", "-1.25"]]
    requests = []
    def sdk_post(**kwargs):
        requests.append(kwargs)
        return {"job_id":"offline-test-job", "job_status":"queued"}
    with patch("triplealpha.deployment.runner.synchronous_runner.rest.post", side_effect=sdk_post):
        sdk = SynchronousRunner(api_url="https://example.invalid/deployment/run/create", api_key="synthetic-only")
        sdk.run_execution({"inputs":values, "num_shots":10})
        sdk.get_job_status("offline-test-job")
        sdk.update_inputs("offline-test-job", values)
        sdk.get_job_results("offline-test-job")
    assert [r["endpoint"].value for r in requests] == ["create", "get_status", "update_input", "get_results"]
    class Response:
        status_code = 200
        def __init__(self, data): self.data = data
        def json(self): return self.data
    calls = []
    def transport(method, url, **kwargs):
        calls.append((method,url,kwargs))
        operation = url.rsplit("/",1)[-1]
        if operation == "create":
            return Response({"job_id":"offline-test-job", "deployment_id":"synthetic", "job_status":"queued",
                             "inputs":values, "missing_inputs":[], "num_shots":10})
        if operation == "get_status":
            return Response({"job_id":"offline-test-job", "job_status":"successful", "missing_inputs":[]})
        if operation == "get_results":
            return Response({"job_id":"offline-test-job", "status":"successful", "results":[]})
        return Response({"job_id":"offline-test-job", "inputs":values})
    # These attestations are synthetic test values, never deployment evidence.
    flags = {key:True for key in ("compiler_verified", "gate_convention_verified", "deployment_verified",
                                  "service_contract_verified", "native_input_binding_verified", "native_output_mapping_verified")}
    client = TripleAlphaClient("https://example.invalid/deployment/run/create", "synthetic-only",
        enable_live=True, native_validation=flags, transport=transport)
    client.create_job(10, inputs=values)
    status = client.get_status("offline-test-job")
    client.update_input("offline-test-job", values)
    results = client.get_results("offline-test-job")
    assert all(call[0] == "POST" for call in calls)
    assert [call[1] for call in calls] == [str(r["api_url"])+"/"+r["endpoint"].value for r in requests]
    # Compare through the actual SDK request models, including optional defaults.
    ours = ExecutionCreateRequest(**calls[0][2]["json"]).model_dump(exclude_none=True)
    expected = {k:v for k,v in requests[0]["json_data"].items() if v is not None}
    assert ours == expected
    assert ExecutionInputsUpdateRequest(**calls[2][2]["json"]).model_dump() == requests[2]["json_data"]
    ExecutionStatusResponse(**status)
    ExecutionResultsResponse(**results)
    assert set(client.KNOWN) == {state.value for state in JobStatus}
    rejected_float_update = False
    try:
        ExecutionInputsUpdateRequest(job_id="offline-test-job", inputs=[["theta_0", .5]])
    except ValueError:
        rejected_float_update = True
    assert rejected_float_update, "Reassess numeric wire typing if the pinned SDK contract changes"
    print(json.dumps({"sdk_version":version("triple-alpha"), "mocked_operations":4,
        "request_models_match":True, "deployment_calls":0,
        "numeric_update_values":"SDK 0.2.3 requires numeric strings",
        "native_compiler_verified":False, "native_output_mapping_verified":False}, indent=2))


if __name__ == "__main__":
    main()
