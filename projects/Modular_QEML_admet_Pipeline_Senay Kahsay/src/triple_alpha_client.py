"""Opt-in HTTP contract pinned to the official triple-alpha 0.2.3 SDK.

Only the request/outer-response contract is established here. Native compiled
input names, gate semantics, deployment identity and output mapping require
separate verification. The default remains offline. Tests inject every request
and clock; no credentials or live API operation are needed.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math
import re
import time
from urllib.parse import urlsplit, urlunsplit

from hamiltonian_validation import finite_real, positive_integer

SDK_CONTRACT_VERSION = "triple-alpha==0.2.3"
_NUMERIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
MAX_NAMED_INPUTS = 1_000_000


class APIContractError(RuntimeError):
    """An HTTP response violates the pinned SDK/local deployment contract."""


class APILifecycleError(RuntimeError):
    """The remote lifecycle cannot complete the required execution."""


def _text(value, name):
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 1024 or any(ord(char) < 32 for char in value)):
        raise ValueError(f"{name} must be a nonempty canonical identifier")
    return value


def _numeric_string(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or _NUMERIC.fullmatch(value) is None:
        raise ValueError("SDK input values must be finite numeric strings")
    try:
        exact, numeric = Decimal(value), float(value)
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("SDK input values must be finite numeric strings") from None
    if not math.isfinite(numeric) or (exact != 0 and numeric == 0):
        raise ValueError("SDK input value overflows or underflows supported numeric precision")
    return value


def validate_named_inputs(inputs, *, allow_empty=False):
    """Validate the named string-pair representation accepted by SDK updates.

    A bare flat numeric stream is deliberately rejected. Its mapping to unique
    compiled input names is a separate deployment binding, not inferred here.
    """
    if not isinstance(allow_empty, bool):
        raise ValueError("allow_empty must be Boolean")
    if not isinstance(inputs, (list, tuple)) or len(inputs) > MAX_NAMED_INPUTS or (not inputs and not allow_empty):
        raise ValueError("SDK inputs require a bounded named name/value sequence")
    result, seen = [], set()
    for item in inputs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each SDK input requires a name and numeric-string value")
        name, value = _text(item[0], "input name"), _numeric_string(item[1])
        if name in seen:
            raise ValueError("SDK input names must be unique within a submission")
        seen.add(name);result.append([name,value])
    return result


def _same_value(left, right):
    return float(left) == float(right)


class TripleAlphaClient:
    # Exact JobStatus values in triplealpha/utils/constants.py, version 0.2.3.
    TERMINAL = {"successful", "failed", "cancelled"}
    KNOWN = TERMINAL | {"queued", "in_progress", "awaiting_input", "awaiting_qpu",
                        "in_flight", "pending_submission"}
    OPERATIONS = {"create", "get_status", "update_input", "get_results"}
    SDK_ENDPOINTS = OPERATIONS | {"cancel", "list_jobs"}
    LIVE_VERIFICATIONS = ("compiler_verified", "gate_convention_verified", "deployment_verified",
                          "service_contract_verified", "native_input_binding_verified",
                          "native_output_mapping_verified")

    def __init__(self, run_url, api_key, *, enable_live=False, native_validation=None,
                 transport=None, clock=time.monotonic, sleep=time.sleep, request_timeout=30.0):
        if not isinstance(run_url, str) or any(char.isspace() for char in run_url):
            raise ValueError("run_url must be an HTTPS deployment API URL")
        parsed = urlsplit(run_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("run_url must be HTTPS without user information, query, or fragment")
        try:
            parsed.port
        except ValueError:
            raise ValueError("run_url has an invalid port") from None
        path = parsed.path.rstrip("/")
        if any(segment in {".", ".."} for segment in path.split("/")) or "%" in path:
            raise ValueError("run_url must use a canonical deployment path")
        # Match SDK Deployment.clean_api_url: remove only a supplied operation.
        # A '/run' segment is preserved when present, never invented when absent.
        if path.rsplit("/",1)[-1] in self.SDK_ENDPOINTS:
            path = path.rsplit("/",1)[0]
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        if not isinstance(api_key, str) or not api_key.strip() or any(char in api_key for char in "\r\n"):
            raise ValueError("A nonempty API key is required for the optional client")
        if not isinstance(enable_live, bool):
            raise ValueError("enable_live must be Boolean")
        self._api_key = api_key
        self.enable_live = enable_live
        self.native_validation = dict(native_validation or {})
        self.transport = transport
        self.clock, self.sleep = clock, sleep
        self.request_timeout = finite_real(request_timeout, "request_timeout")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

    def __repr__(self):
        return f"TripleAlphaClient(base_url={self.base_url!r}, enable_live={self.enable_live!r})"

    def endpoint(self, operation):
        if operation not in self.OPERATIONS:
            raise ValueError("Unsupported API operation")
        return f"{self.base_url}/{operation}"

    def _ensure_authorized(self):
        if not self.enable_live:
            raise RuntimeError("Live execution is disabled; preprocessing and handoff are offline")
        if any(self.native_validation.get(key) is not True for key in self.LIVE_VERIFICATIONS):
            raise RuntimeError("Live execution requires verified compiler, gates, deployment, SDK service contract, native input binding and output mapping")

    @staticmethod
    def _job_id(job_id):
        return _text(job_id,"job_id")

    def _request(self, operation, body, *, deadline=None):
        timeout = self.request_timeout
        if deadline is not None:
            deadline = finite_real(deadline,"deadline")
            remaining = deadline-finite_real(self.clock(),"clock value")
            if remaining <= 0:
                raise TimeoutError("API lifecycle deadline exceeded")
            timeout = min(timeout,remaining)
        self._ensure_authorized()
        request = self.transport
        if request is None:
            import requests
            request = requests.request
        try:
            response = request("POST",self.endpoint(operation),
                headers={"X-Api-Key":self._api_key,"Content-Type":"application/json"},
                json=body,timeout=timeout,allow_redirects=False)
        except Exception:
            raise APIContractError(f"Transport failed during {operation}") from None
        # Preserve a possibly-created remote job's verified receipt even when
        # CREATE returns late. Other operations already have a known job ID.
        if operation != "create" and deadline is not None and finite_real(self.clock(),"clock value") >= deadline:
            raise TimeoutError("API lifecycle deadline exceeded")
        status = getattr(response,"status_code",None)
        if isinstance(status,bool) or not isinstance(status,int) or not 200 <= status < 300:
            raise APIContractError(f"HTTP request failed during {operation}")
        try:
            data = response.json()
        except Exception:
            raise APIContractError(f"Invalid JSON response during {operation}") from None
        if not isinstance(data,dict):
            raise APIContractError(f"Expected a JSON object during {operation}")
        if "job_id" in body and data.get("job_id") != body["job_id"]:
            raise APIContractError("Response job identifier mismatch")
        return data

    def _status(self,data,field):
        if not isinstance(data.get(field),str) or data[field] not in self.KNOWN:
            raise APIContractError("Response contained an unknown SDK lifecycle state")
        return data[field]

    def create_job(self,num_shots,*,inputs=None,api_call_id=None,deadline=None):
        """Return a verified creation receipt even if its response arrives late.

        The request timeout is bounded by deadline. A late known job ID must
        not be discarded or automatically recreated; callers persist the
        returned receipt and enforce their remaining lifecycle budget.
        """
        positive_integer(num_shots,"num_shots")
        named = validate_named_inputs([] if inputs is None else inputs,allow_empty=True)
        if api_call_id is not None:
            _text(api_call_id,"api_call_id")
        body = {"inputs":named,"num_shots":int(num_shots)}
        if api_call_id is not None:
            body["api_call_id"] = api_call_id
        data = self._request("create",body,deadline=deadline)
        try:
            self._job_id(data.get("job_id"));_text(data.get("deployment_id"),"deployment_id")
            shots = positive_integer(data.get("num_shots"),"response num_shots")
            if shots != num_shots:raise ValueError("shot count changed")
            for key in ("inputs","missing_inputs"):
                rows = data.get(key,[])
                if not isinstance(rows,list) or len(rows)>MAX_NAMED_INPUTS or any(not isinstance(row,list) for row in rows):
                    raise ValueError("invalid create input rows")
        except ValueError:
            raise APIContractError("Create response omitted valid SDK job metadata") from None
        self._status(data,"job_status")
        return data

    def get_status(self,job_id,*,deadline=None):
        data = self._request("get_status",{"job_id":self._job_id(job_id)},deadline=deadline)
        self._status(data,"job_status")
        missing = data.get("missing_inputs")
        if not isinstance(missing,list) or len(missing)>MAX_NAMED_INPUTS:
            raise APIContractError("Status response omitted SDK missing-input names")
        try:
            names = [_text(value,"missing input name") for value in missing]
        except ValueError:
            raise APIContractError("Status response contained malformed missing-input names") from None
        if len(set(names)) != len(names):
            raise APIContractError("Status response repeated a missing-input name")
        return data

    def update_input(self,job_id,inputs,*,deadline=None):
        named = validate_named_inputs(inputs)
        data = self._request("update_input",{"job_id":self._job_id(job_id),"inputs":named},deadline=deadline)
        try:
            returned = dict(validate_named_inputs(data.get("inputs")))
        except ValueError:
            raise APIContractError("Input-update response differs from SDK named string pairs") from None
        if any(name not in returned or not _same_value(value,returned[name]) for name,value in named):
            raise APIContractError("Input-update response does not acknowledge the supplied values")
        return data

    def get_results(self,job_id,*,deadline=None):
        data = self._request("get_results",{"job_id":self._job_id(job_id)},deadline=deadline)
        # ExecutionResultsResponse uses 'status'; create/status use 'job_status'.
        if self._status(data,"status") != "successful":
            raise APIContractError("Results response was not successful")
        results = data.get("results")
        if (results is not None and not isinstance(results,(dict,list))) or (
                isinstance(results,list) and any(not isinstance(row,dict) for row in results)):
            raise APIContractError("Results response differs from the SDK outer result schema")
        return data

    def wait_for(self,job_id,accepted,*,timeout=1800.0,poll_interval=8.0,deadline=None):
        timeout,interval = finite_real(timeout,"timeout"),finite_real(poll_interval,"poll_interval")
        if timeout <= 0 or interval <= 0:raise ValueError("Polling timeout and interval must be positive")
        try:accepted=set(accepted)
        except (TypeError,ValueError):raise ValueError("accepted must contain known SDK states") from None
        if not accepted or not accepted <= self.KNOWN:raise ValueError("accepted must contain known SDK states")
        if deadline is None:deadline=finite_real(self.clock(),"clock value")+timeout
        deadline=finite_real(deadline,"deadline")
        while True:
            data=self.get_status(job_id,deadline=deadline);state=data["job_status"]
            if state in accepted:return data
            if state in self.TERMINAL:raise APILifecycleError(f"Job reached terminal state {state} before the required state")
            remaining=deadline-finite_real(self.clock(),"clock value")
            if remaining<=0:raise TimeoutError("API lifecycle deadline exceeded")
            self.sleep(min(interval,remaining))

    def execute(self,inputs,*,num_shots=10000,timeout=1800.0,poll_interval=8.0,api_call_id=None,on_created=None):
        """Submit complete named inputs; honor only explicit known missing names.

        Repeated or unknown dynamic input requests fail rather than guessing a
        compiler's variable queue. A single absolute deadline covers all stages.
        Persisted job/task receipts are the caller's responsibility.
        """
        if on_created is not None and not callable(on_created):
            raise ValueError("on_created must be a callable receipt-persistence hook")
        named=validate_named_inputs(inputs)
        timeout,interval=finite_real(timeout,"timeout"),finite_real(poll_interval,"poll_interval")
        if timeout<=0 or interval<=0:raise ValueError("Polling timeout and interval must be positive")
        deadline=finite_real(finite_real(self.clock(),"clock value")+timeout,"deadline")
        created=self.create_job(num_shots,inputs=named,api_call_id=api_call_id,deadline=deadline)
        job_id=created["job_id"]
        if on_created is not None:
            try:
                on_created(deepcopy(created))
            except Exception:
                safe_id = "[redacted]" if self._api_key in job_id else job_id
                raise APIContractError(f"Receipt persistence failed for created job {safe_id}; resume it rather than recreate") from None
        if finite_real(self.clock(),"clock value") >= deadline:
            safe_id = "[redacted]" if self._api_key in job_id else job_id
            raise TimeoutError(f"API lifecycle deadline exceeded after creating job {safe_id}; resume it rather than recreate")
        if created["job_status"] in {"failed","cancelled"}:
            raise APILifecycleError("Created job reached a terminal failure state")
        supplied,updated=dict(named),set()
        while True:
            data=self.wait_for(job_id,{"successful","awaiting_input"},deadline=deadline,poll_interval=interval)
            if data["job_status"]=="successful":return self.get_results(job_id,deadline=deadline)
            missing=data["missing_inputs"]
            if not missing or any(name not in supplied or name in updated for name in missing):
                raise APIContractError("Native job requested unknown, empty, or repeated input names")
            response=self.update_input(job_id,[[name,supplied[name]] for name in missing],deadline=deadline)
            if any(name not in supplied or not _same_value(value,supplied[name]) for name,value in response["inputs"]):
                raise APIContractError("Native update acknowledged values outside the declared input binding")
            updated.update(missing)
