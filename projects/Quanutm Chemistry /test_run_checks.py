"""Protect study selection, source files and review publication during execution."""
import os
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_checks


class RunnerTests(unittest.TestCase):
    def test_manifest_covers_the_three_standalone_studies(self):
        self.assertEqual(set(run_checks.STUDIES), {'h2', 'hcn', 'scaling'})
        for study in run_checks.STUDIES.values():
            self.assertTrue((run_checks.ROOT/study.notebook).is_file())
            self.assertTrue((run_checks.ROOT/(study.test_module+'.py')).is_file())

    def test_smoke_profile_cannot_replace_review_evidence(self):
        environment = {**os.environ, 'HCN_PROFILE': 'smoke'}
        result = subprocess.run([sys.executable, str(Path(run_checks.__file__)),
                                 '--study', 'hcn', '--save-review'],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('default profiles', result.stderr)

    def test_generated_outputs_cannot_target_publication_inputs(self):
        for destination in ['.', 'benchmarks/h2', 'benchmarks/scaling']:
            with patch.dict(os.environ, H2_OUTPUT_DIR=destination):
                with self.assertRaises(ValueError):
                    run_checks.output_directory(run_checks.STUDIES['h2'])

    def test_execution_failure_does_not_replace_saved_notebook_or_snapshot(self):
        study = run_checks.STUDIES['h2']
        notebook = run_checks.ROOT/study.notebook
        snapshot = run_checks.ROOT/'benchmarks/h2/results.json'
        original = notebook.read_bytes(), snapshot.read_bytes()
        with tempfile.TemporaryDirectory() as output:
            with patch('run_checks.subprocess.run'):
                with patch('run_checks.execute_study', side_effect=[(None, Path(output)), RuntimeError('kernel failed')]):
                    with patch('run_checks.publish_review') as publish:
                        with redirect_stdout(io.StringIO()):
                            with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                                run_checks.main(['--save-review'])
                        publish.assert_not_called()
        self.assertEqual((notebook.read_bytes(), snapshot.read_bytes()), original)


if __name__ == '__main__':
    unittest.main()
