"""Injected transport tests for official triple-alpha 0.2.3 request semantics."""
import unittest

from test_runtime_fixtures import payload, settings
from runtime_input_protocol import build_runtime_input_values
from triple_alpha_client import (TripleAlphaClient, APIContractError, APILifecycleError,
                                 validate_named_inputs, SDK_CONTRACT_VERSION)

VERIFIED_TEST_ONLY = {name:True for name in TripleAlphaClient.LIVE_VERIFICATIONS}


def named_values():
    return [[f'slot_{i:05d}',format(value,'.17g')]
            for i,value in enumerate(build_runtime_input_values(payload(),settings()))]


def created(*,job='job-1',shots=50,status='queued'):
    return {'job_id':job,'deployment_id':'synthetic-deployment','job_status':status,
            'inputs':[],'missing_inputs':[],'num_shots':shots}


def state(value,*,job='job-1',missing=None):
    return {'job_id':job,'job_status':value,'missing_inputs':[] if missing is None else missing}


class Response:
    def __init__(self,data,status=200):self.data,self.status_code=data,status
    def json(self):
        if isinstance(self.data,Exception):raise self.data
        return self.data


class Transport:
    def __init__(self,responses):self.responses,self.calls=list(responses),[]
    def __call__(self,method,url,**kwargs):
        self.calls.append((method,url,kwargs));result=self.responses.pop(0)
        if isinstance(result,Exception):raise result
        return result


class Clock:
    def __init__(self):self.now,self.sleeps=0.,[]
    def __call__(self):return self.now
    def sleep(self,duration):self.sleeps.append(duration);self.now+=duration


def client(responses,**kwargs):
    transport,clock=Transport(responses),Clock()
    result=TripleAlphaClient('https://example.invalid/deployment/run/create','synthetic-secret',
        enable_live=True,native_validation=VERIFIED_TEST_ONLY,transport=transport,
        clock=clock,sleep=clock.sleep,**kwargs)
    return result,transport,clock


class APIClientTests(unittest.TestCase):
    def test_complete_initial_inputs_lifecycle_matches_sdk_fields_and_all_post(self):
        c,transport,clock=client([Response(created()),
            *[Response(state(s)) for s in ['queued','in_progress','awaiting_qpu','in_flight','successful']],
            Response({'job_id':'job-1','status':'successful','results':{'native_output':'opaque'}})])
        named=named_values()
        result=c.execute(named,num_shots=50,timeout=30,poll_interval=2,api_call_id='synthetic-call')
        self.assertEqual(result['results'],{'native_output':'opaque'})
        self.assertTrue(all(call[0]=='POST' for call in transport.calls))
        self.assertEqual(transport.calls[0][2]['json'],{'api_call_id':'synthetic-call','inputs':named,'num_shots':50})
        self.assertTrue(all(call[2]['allow_redirects'] is False for call in transport.calls))
        self.assertEqual(clock.sleeps,[2]*4)
        self.assertNotIn('synthetic-secret',repr(c))
        self.assertEqual(SDK_CONTRACT_VERSION,'triple-alpha==0.2.3')

    def test_explicit_missing_name_update_uses_named_string_pairs_and_post(self):
        named=named_values();first=named[0]
        c,transport,_=client([Response(created()),Response(state('awaiting_input',missing=[first[0]])),
            Response({'job_id':'job-1','inputs':[first]}),Response(state('successful')),
            Response({'job_id':'job-1','status':'successful','results':[]})])
        c.execute(named,num_shots=50)
        self.assertEqual([c[1].rsplit('/',1)[-1] for c in transport.calls],
                         ['create','get_status','update_input','get_status','get_results'])
        self.assertEqual(transport.calls[2][0],'POST')
        self.assertEqual(transport.calls[2][2]['json'],{'job_id':'job-1','inputs':[first]})
        self.assertTrue(all(isinstance(v,str) for pair in transport.calls[2][2]['json']['inputs'] for v in pair))

    def test_unknown_empty_or_repeated_native_input_queue_is_not_guessed(self):
        named=named_values();first=named[0]
        for missing in ([],['unknown-native-variable']):
            c,t,_=client([Response(created()),Response(state('awaiting_input',missing=missing))])
            with self.assertRaises(APIContractError):c.execute(named,num_shots=50)
            self.assertEqual(len(t.calls),2)
        c,t,_=client([Response(created()),Response(state('awaiting_input',missing=[first[0]])),
            Response({'job_id':'job-1','inputs':[first]}),Response(state('awaiting_input',missing=[first[0]]))])
        with self.assertRaisesRegex(APIContractError,'repeated'):c.execute(named,num_shots=50)
        self.assertEqual(len(t.calls),4)

    def test_endpoint_cleanup_matches_sdk_and_never_invents_run_segment(self):
        for base in ('https://example.invalid/deployment/run','https://example.invalid/deployment/run/',
                     'https://example.invalid/deployment/run/create','https://example.invalid/deployment/run/get_status',
                     'https://example.invalid/deployment/run/update_input','https://example.invalid/deployment/run/get_results'):
            self.assertEqual(TripleAlphaClient(base,'synthetic').endpoint('create'),
                             'https://example.invalid/deployment/run/create')
        self.assertEqual(TripleAlphaClient('https://example.invalid/deployment','synthetic').endpoint('create'),
                         'https://example.invalid/deployment/create')
        for base in ('http://example.invalid','https://user:pass@example.invalid','https://example.invalid?q=x',
                     'https://example.invalid#x','https://example.invalid/a/../b','https://example.invalid/%2e%2e/b'):
            with self.subTest(base=base),self.assertRaises(ValueError):TripleAlphaClient(base,'synthetic')

    def test_live_gate_requires_all_native_binding_and_mapping_verifications(self):
        transport=Transport([])
        for enabled,flags in [(False,VERIFIED_TEST_ONLY),(True,{})]+[(True,{**VERIFIED_TEST_ONLY,k:False}) for k in VERIFIED_TEST_ONLY]:
            c=TripleAlphaClient('https://example.invalid/run/create','synthetic',enable_live=enabled,native_validation=flags,transport=transport)
            with self.assertRaises(RuntimeError):c.create_job(1)
        self.assertEqual(transport.calls,[])

    def test_named_input_type_finiteness_and_unique_name_checks_precede_transport(self):
        c,t,_=client([])
        bad=[[],[1.,2.],[['x',1.]], [['x','1'],['x','2']], [['','1']], [['x','NaN']],
             [['x','Infinity']], [['x','1e999']], [['x','1e-999']], [['x','1_000']], [['x',' 1 ']], [['x','true']]]
        for values in bad:
            with self.subTest(values=values),self.assertRaises(ValueError):c.execute(values)
        for shots in (0,-1,True,1.5):
            with self.assertRaises(ValueError):c.create_job(shots)
        for timeout,interval in ((0,1),(1,0),(float('nan'),1),(1,float('inf'))):
            with self.assertRaises(ValueError):c.execute(named_values(),timeout=timeout,poll_interval=interval)
        self.assertEqual(t.calls,[])
        self.assertEqual(validate_named_inputs([['x','1e-300']]),[['x','1e-300']])

    def test_terminal_failures_and_unknown_sdk_states_are_rejected(self):
        for terminal in ('failed','cancelled'):
            c,t,_=client([Response(created()),Response(state(terminal))])
            with self.assertRaises(APILifecycleError):c.execute(named_values(),num_shots=50)
            self.assertEqual(len(t.calls),2)
        for invalid in ('running','created','canceled','unexpected'):
            c,_,_=client([Response(state(invalid))])
            with self.assertRaises(APIContractError):c.get_status('job-1')
        self.assertEqual(TripleAlphaClient.KNOWN,{'queued','in_progress','successful','failed','cancelled',
                                               'awaiting_input','awaiting_qpu','in_flight','pending_submission'})

    def test_timeout_uses_injected_clock_and_remaining_deadline(self):
        c,t,clock=client([Response(state('queued')) for _ in range(3)])
        with self.assertRaises(TimeoutError):c.wait_for('job-1',{'successful'},timeout=3,poll_interval=2)
        self.assertEqual(clock.sleeps,[2,1])
        self.assertEqual([call[2]['timeout'] for call in t.calls],[3,1])
        for invalid in (float('nan'),float('inf'),True):
            c,t,_=client([])
            with self.assertRaises(ValueError):c.create_job(1,deadline=invalid)
            with self.assertRaises(ValueError):c.wait_for('job-1',{'successful'},deadline=invalid)
            self.assertEqual(t.calls,[])

    def test_results_use_status_field_and_errors_do_not_echo_keys_or_server_body(self):
        secret='synthetic-secret'
        cases=[Response({'private':secret},401),Response({'private':secret},302),Response(ValueError(secret)),
               Response([secret]),RuntimeError(secret),Response(state('unknown')),Response(state('queued',job='wrong'))]
        for response in cases:
            c,_,_=client([response])
            with self.assertRaises(APIContractError) as caught:c.get_status('job-1')
            self.assertNotIn(secret,str(caught.exception))
        c,_,_=client([Response({'job_id':'job-1','job_status':'successful','results':[]})])
        with self.assertRaises(APIContractError):c.get_results('job-1')
        c,_,_=client([Response({'job_id':'job-1','status':'successful','results':[]})])
        self.assertEqual(c.get_results('job-1')['status'],'successful')
        c,_,_=client([Response({'job_id':'job-1'})])
        with self.assertRaises(APIContractError):c.create_job(50)

    def test_receipt_hook_runs_before_polling_and_failure_stops_requests(self):
        c,t,_=client([Response(created()),Response(state('successful')),
            Response({'job_id':'job-1','status':'successful','results':[]})])
        receipts=[]
        def persist(response):
            self.assertEqual(len(t.calls),1)
            receipts.append(response['job_id'])
            response['job_id']='mutated-copy'
        c.execute(named_values(),num_shots=50,on_created=persist)
        self.assertEqual(receipts,['job-1'])
        c,t,_=client([Response(created())])
        def broken(response):raise RuntimeError('synthetic-secret')
        with self.assertRaises(APIContractError) as caught:c.execute(named_values(),num_shots=50,on_created=broken)
        self.assertIn('job-1',str(caught.exception));self.assertNotIn('synthetic-secret',str(caught.exception))
        self.assertEqual(len(t.calls),1)
        c,t,_=client([])
        with self.assertRaises(ValueError):c.execute(named_values(),on_created=7)
        self.assertEqual(t.calls,[])

    def test_late_create_persists_known_job_before_deadline_failure(self):
        clock=Clock();calls=[];receipts=[]
        def late_create(*args,**kwargs):
            calls.append(kwargs['json']);clock.now=5
            return Response(created())
        c=TripleAlphaClient('https://example.invalid/run/create','synthetic',enable_live=True,
            native_validation=VERIFIED_TEST_ONLY,transport=late_create,clock=clock,sleep=clock.sleep)
        with self.assertRaises(TimeoutError) as caught:
            c.execute(named_values(),num_shots=50,timeout=3,on_created=lambda value:receipts.append(value['job_id']))
        self.assertEqual(receipts,['job-1'])
        self.assertIn('job-1',str(caught.exception))
        self.assertEqual(len(calls),1)
        # The low-level creation API exposes the known receipt to its caller
        # rather than losing the job ID merely because the response was late.
        clock.now=0
        self.assertEqual(c.create_job(50,inputs=named_values(),deadline=3)['job_id'],'job-1')
        self.assertEqual(len(calls),2)

    def test_response_exhausting_deadline_cannot_complete_lifecycle(self):
        clock=Clock()
        def late(*args,**kwargs):clock.now=5;return Response(state('successful'))
        c=TripleAlphaClient('https://example.invalid/run/create','synthetic',enable_live=True,
            native_validation=VERIFIED_TEST_ONLY,transport=late,clock=clock,sleep=clock.sleep)
        with self.assertRaises(TimeoutError):c.wait_for('job-1',{'successful'},timeout=3)


if __name__=='__main__':unittest.main()
