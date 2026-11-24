"""Opt-in adapter for a provisional Triple Alpha deployment contract.

The endpoint names and PATCH update method follow the supplied project notebook;
public vendor material does not establish their schema. The adapter therefore
fails closed until compiler, native gate conventions, and the service contract
have separately been verified. Tests inject a transport and a monotonic clock;
no request is issued by construction or by offline preprocessing.
"""
from __future__ import annotations

import time
from urllib.parse import urlsplit, urlunsplit

from hamiltonian_validation import finite_real, positive_integer
from runtime_input_protocol import decode_runtime_input_values


class APIContractError(RuntimeError):
    """An HTTP response does not satisfy the configured local API contract."""


class APILifecycleError(RuntimeError):
    """The remote lifecycle ended without reaching the required state."""


class TripleAlphaClient:
    TERMINAL = {"successful", "failed", "cancelled", "canceled"}
    KNOWN = TERMINAL | {"created", "queued", "running", "awaiting_input"}
    OPERATIONS = {"create", "get_status", "update_input", "get_results"}

    def __init__(self, run_url, api_key, *, enable_live=False, native_validation=None,
                 transport=None, clock=time.monotonic, sleep=time.sleep, request_timeout=30.0):
        if not isinstance(run_url, str) or any(char.isspace() for char in run_url):
            raise ValueError("run_url must be an HTTPS deployment URL")
        parsed = urlsplit(run_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("run_url must be HTTPS without user information, query, or fragment")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("run_url has an invalid port") from exc
        path = parsed.path.rstrip("/")
        if any(segment in {".", ".."} for segment in path.split("/")) or "%" in path:
            raise ValueError("run_url must use a canonical deployment path")
        if path.endswith("/run"):
            path = path[:-4]
        if path.endswith(tuple("/" + value for value in self.OPERATIONS)):
            raise ValueError("run_url must name the deployment base, not an operation")
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        if not isinstance(api_key, str) or not api_key.strip() or any(char in api_key for char in "\r\n"):
            raise ValueError("A nonempty API key is required for the optional client")
        if not isinstance(enable_live, bool):
            raise ValueError("enable_live must be a Boolean")
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
        return f"{self.base_url}/run/{operation}"

    def _ensure_authorized(self):
        if not self.enable_live:
            raise RuntimeError("Live execution is disabled; preprocessing and handoff are offline")
        required = ("compiler_verified", "gate_convention_verified", "deployment_verified", "service_contract_verified")
        if any(self.native_validation.get(key) is not True for key in required):
            raise RuntimeError("Live execution requires verified compiler, gate conventions, deployment, and service contract")

    @staticmethod
    def _job_id(job_id):
        if not isinstance(job_id, str) or not job_id.strip() or job_id != job_id.strip():
            raise ValueError("job_id must be a nonempty identifier")
        return job_id

    def _request(self, operation, body, *, deadline=None):
        self._ensure_authorized()
        timeout = self.request_timeout
        if deadline is not None:
            deadline = finite_real(deadline, "deadline")
            remaining = deadline - finite_real(self.clock(), "clock value")
            if remaining <= 0:
                raise TimeoutError("API lifecycle deadline exceeded")
            timeout = min(timeout, remaining)
        request = self.transport
        if request is None:
            import requests
            request = requests.request
        method = "PATCH" if operation == "update_input" else "POST"
        try:
            response = request(method, self.endpoint(operation),
                               headers={"X-Api-Key": self._api_key, "Content-Type": "application/json"},
                               json=body, timeout=timeout, allow_redirects=False)
        except Exception:
            raise APIContractError(f"Transport failed during {operation}") from None
        if deadline is not None and finite_real(self.clock(), "clock value") >= deadline:
            raise TimeoutError("API lifecycle deadline exceeded")
        status = getattr(response, "status_code", None)
        if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
            raise APIContractError(f"HTTP request failed during {operation}")
        try:
            data = response.json()
        except Exception:
            raise APIContractError(f"Invalid JSON response during {operation}") from None
        if not isinstance(data, dict):
            raise APIContractError(f"Expected a JSON object during {operation}")
        if "job_id" in body and "job_id" in data and data["job_id"] != body["job_id"]:
            raise APIContractError("Response job identifier mismatch")
        return data

    def create_job(self, num_shots, *, deadline=None):
        positive_integer(num_shots, "num_shots")
        data = self._request("create", {"num_shots": int(num_shots)}, deadline=deadline)
        try:
            self._job_id(data.get("job_id"))
        except ValueError:
            raise APIContractError("Create response omitted a valid job identifier") from None
        return data

    def get_status(self, job_id, *, deadline=None):
        data = self._request("get_status", {"job_id": self._job_id(job_id)}, deadline=deadline)
        if not isinstance(data.get("job_status"), str) or data["job_status"] not in self.KNOWN:
            raise APIContractError("Status response contained an unknown lifecycle state")
        return data

    def update_input(self, job_id, inputs, *, deadline=None):
        decode_runtime_input_values(inputs)
        return self._request("update_input", {"job_id": self._job_id(job_id), "inputs": list(inputs)}, deadline=deadline)

    def get_results(self, job_id, *, deadline=None):
        data = self._request("get_results", {"job_id": self._job_id(job_id)}, deadline=deadline)
        if data.get("job_status") != "successful":
            raise APIContractError("Results response was not successful")
        return data

    def wait_for(self, job_id, accepted, *, timeout=1800.0, poll_interval=8.0, deadline=None):
        timeout = finite_real(timeout, "timeout")
        interval = finite_real(poll_interval, "poll_interval")
        if timeout <= 0 or interval <= 0:
            raise ValueError("Polling timeout and interval must be positive")
        accepted = set(accepted)
        if not accepted or not accepted <= self.KNOWN:
            raise ValueError("accepted must contain known lifecycle states")
        if deadline is None:
            deadline = finite_real(self.clock(), "clock value") + timeout
        deadline = finite_real(deadline, "deadline")
        while True:
            data = self.get_status(job_id, deadline=deadline)
            state = data["job_status"]
            if state in accepted:
                return data
            if state in self.TERMINAL:
                raise APILifecycleError(f"Job reached terminal state {state} before the required state")
            remaining = deadline - finite_real(self.clock(), "clock value")
            if remaining <= 0:
                raise TimeoutError("API lifecycle deadline exceeded")
            self.sleep(min(interval, remaining))

    def execute(self, inputs, *, num_shots=10000, timeout=1800.0, poll_interval=8.0):
        """Run the provisional lifecycle with one shared bounded deadline."""
        decode_runtime_input_values(inputs)
        timeout = finite_real(timeout, "timeout")
        interval = finite_real(poll_interval, "poll_interval")
        if timeout <= 0 or interval <= 0:
            raise ValueError("Polling timeout and interval must be positive")
        deadline = finite_real(finite_real(self.clock(), "clock value") + timeout, "deadline")
        job_id = self.create_job(num_shots, deadline=deadline)["job_id"]
        self.wait_for(job_id, {"awaiting_input"}, deadline=deadline, poll_interval=interval)
        self.update_input(job_id, inputs, deadline=deadline)
        self.wait_for(job_id, {"successful"}, deadline=deadline, poll_interval=interval)
        return self.get_results(job_id, deadline=deadline)
