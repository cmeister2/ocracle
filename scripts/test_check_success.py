"""Aggregate checks must not hide failed, cancelled or skipped dependencies."""

import unittest
from pathlib import Path
import re

from check_success import REQUIRED_JOBS, check


def successful_results():
    return {job: {'result': 'success'} for job in REQUIRED_JOBS}


class SuccessGate(unittest.TestCase):
    def test_all_required_jobs_succeed(self):
        self.assertEqual(check(successful_results()), {})

    def test_each_non_success_result_blocks_the_gate(self):
        for job in REQUIRED_JOBS:
            for result in ['failure', 'cancelled', 'skipped', 'unknown']:
                with self.subTest(job=job, result=result):
                    results = successful_results()
                    results[job] = {'result': result}
                    self.assertEqual(check(results), {job: result})

    def test_each_missing_job_blocks_the_gate(self):
        for job in REQUIRED_JOBS:
            with self.subTest(job=job):
                results = successful_results()
                del results[job]
                self.assertEqual(check(results), {job: 'missing job'})

    def test_unexpected_job_fails_the_explicit_contract(self):
        results = successful_results()
        results['unregistered'] = {'result': 'success'}
        self.assertEqual(check(results), {'unregistered': 'unexpected job'})

    def test_absent_or_malformed_results_fail_closed(self):
        for result in [None, {}, []]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                check(result)
        for details in [{}, None, [], {'result': None}]:
            with self.subTest(details=details):
                results = successful_results()
                results['native-training'] = details
                self.assertEqual(check(results), {'native-training': 'missing result'})

    def test_workflow_needs_matches_the_required_job_contract(self):
        workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text()
        success = workflow.split('\n  success:\n', 1)[1].split('\n  release:\n', 1)[0]
        needs = re.search(r'^    needs: \[([^\]]+)\]$', success, re.MULTILINE)
        self.assertIsNotNone(needs, 'Keep the aggregate dependencies explicit')
        jobs = [job.strip() for job in needs.group(1).split(',')]
        self.assertEqual(len(jobs), len(set(jobs)), 'Aggregate job dependencies must be unique')
        self.assertEqual(frozenset(jobs), REQUIRED_JOBS)
        defined_jobs = set(re.findall(r'^  ([a-z][a-z0-9-]*):$', workflow, re.MULTILINE))
        self.assertTrue(REQUIRED_JOBS <= defined_jobs)


if __name__ == '__main__':
    unittest.main()
