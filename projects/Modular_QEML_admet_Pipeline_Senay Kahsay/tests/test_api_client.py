import unittest

from test_runtime_fixtures import payload, settings
from runtime_input_protocol import build_runtime_input_values
from triple_alpha_client import TripleAlphaClient, APIContractError, APILifecycleError

VERIFIED_TEST_ONLY = {"compiler_verified": True, "gate_convention_verified": True,
                      "deployment_verified": True, "service_contract_verified": True}


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class Transport:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def client(responses, **kwargs):
    transport, clock = Transport(responses), Clock()
    result = TripleAlphaClient("https://example.invalid/deployment/run/", "synthetic-secret",
                               enable_live=True, native_validation=VERIFIED_TEST_ONLY,
                               transport=transport, clock=clock, sleep=clock.sleep, **kwargs)
    return result, transport, clock


class APIClientTests(unittest.TestCase):
    def test_complete_lifecycle_uses_one_contract_without_real_network(self):
        c, transport, clock = client([
            Response({"job_id": "job-1"}),
            Response({"job_status": "queued", "job_id": "job-1"}),
            Response({"job_status": "awaiting_input"}), Response({"accepted": True}),
            Response({"job_status": "running"}), Response({"job_status": "successful"}),
            Response({"job_status": "successful", "results": [[0, 0]]})])
        values = build_runtime_input_values(payload(), settings())
        result = c.execute(values, num_shots=50, timeout=30, poll_interval=2)
        self.assertEqual(result["results"], [[0, 0]])
        operations = [call[1].rsplit("/", 1)[-1] for call in transport.calls]
        self.assertEqual(operations, ["create", "get_status", "get_status", "update_input",
                                      "get_status", "get_status", "get_results"])
        self.assertEqual(transport.calls[3][0], "PATCH")
        self.assertTrue(all(call[0] == "POST" for index, call in enumerate(transport.calls) if index != 3))
        self.assertEqual(transport.calls[3][2]["json"]["inputs"], values)
        self.assertTrue(all(call[2]["allow_redirects"] is False for call in transport.calls))
        self.assertEqual(clock.sleeps, [2, 2])
        self.assertEqual(transport.calls[0][2]["json"], {"num_shots": 50})
        self.assertNotIn("synthetic-secret", repr(c))

    def test_endpoint_base_variants_and_rejected_urls(self):
        for base in ("https://example.invalid/deployment", "https://example.invalid/deployment/",
                     "https://example.invalid/deployment/run", "https://example.invalid/deployment/run/"):
            c = TripleAlphaClient(base, "synthetic")
            self.assertEqual(c.endpoint("update_input"), "https://example.invalid/deployment/run/update_input")
        for base in ("http://example.invalid", "https://user:pass@example.invalid", "https://example.invalid?q=x",
                     "https://example.invalid#x", "https://example.invalid/run/update_input",
                     "https://example.invalid/a/../b", "https://example.invalid/%2e%2e/b"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                TripleAlphaClient(base, "synthetic")

    def test_live_execution_requires_explicit_opt_in_and_all_verifications(self):
        transport = Transport([])
        for enabled, flags in ((False, VERIFIED_TEST_ONLY), (True, {}),
                               (True, {"compiler_verified": True}),
                               (True, {**VERIFIED_TEST_ONLY, "service_contract_verified": False})):
            c = TripleAlphaClient("https://example.invalid", "synthetic", enable_live=enabled,
                                   native_validation=flags, transport=transport)
            with self.assertRaises(RuntimeError):
                c.create_job(1)
        self.assertEqual(transport.calls, [])

    def test_nonfinite_or_invalid_inputs_never_reach_transport(self):
        c, transport, _ = client([])
        for shots in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                c.create_job(shots)
        with self.assertRaises(ValueError):
            c.update_input("job-1", [float("nan")] * 5)
        with self.assertRaises(ValueError):
            c.get_status(" ")
        for timeout, interval in ((0, 1), (1, 0), (float("nan"), 1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                c.execute(build_runtime_input_values(payload(), settings()), timeout=timeout, poll_interval=interval)
        self.assertEqual(transport.calls, [])

    def test_terminal_failure_prevents_input_update(self):
        for terminal in ("failed", "canceled", "cancelled", "successful"):
            c, transport, _ = client([Response({"job_id": "job-1"}), Response({"job_status": terminal})])
            with self.assertRaises(APILifecycleError):
                c.execute(build_runtime_input_values(payload(), settings()))
            self.assertEqual(len(transport.calls), 2)

    def test_timeout_uses_injected_clock_and_remaining_deadline(self):
        c, transport, clock = client([Response({"job_status": "queued"}) for _ in range(3)])
        with self.assertRaises(TimeoutError):
            c.wait_for("job-1", {"awaiting_input"}, timeout=3, poll_interval=2)
        self.assertEqual(clock.sleeps, [2, 1])
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual([call[2]["timeout"] for call in transport.calls], [3, 1])

    def test_error_responses_do_not_echo_keys_or_server_body(self):
        secret = "synthetic-secret"
        cases = [Response({"private": secret}, 401), Response({"private": secret}, 302),
                 Response(ValueError(secret)), Response([secret]), RuntimeError(secret),
                 Response({"job_status": {"unexpected": "object"}}),
                 Response({"job_status": "unrecognized", "private": secret}),
                 Response({"job_status": "running", "job_id": "wrong", "private": secret})]
        for response in cases:
            c, _, _ = client([response])
            with self.subTest(response=response), self.assertRaises(APIContractError) as caught:
                c.get_status("job-1")
            self.assertNotIn(secret, str(caught.exception))
        c, _, _ = client([Response({"private": secret})])
        with self.assertRaises(APIContractError):
            c.create_job(1)
        c, _, _ = client([Response({"job_status": "running"})])
        with self.assertRaises(APIContractError):
            c.get_results("job-1")

    def test_response_that_exhausts_deadline_cannot_complete_lifecycle(self):
        clock = Clock()
        def late_response(*args, **kwargs):
            clock.now = 5
            return Response({"job_status": "successful"})
        c = TripleAlphaClient("https://example.invalid", "synthetic", enable_live=True,
                              native_validation=VERIFIED_TEST_ONLY, transport=late_response,
                              clock=clock, sleep=clock.sleep)
        with self.assertRaises(TimeoutError):
            c.wait_for("job-1", {"successful"}, timeout=3)


if __name__ == "__main__":
    unittest.main()
