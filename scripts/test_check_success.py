"""Aggregate checks must not hide failed, cancelled or skipped dependencies."""

import unittest

from check_success import check


class SuccessGate(unittest.TestCase):
    def test_all_required_jobs_succeed(self):
        self.assertEqual(check({'rust': {'result': 'success'}, 'neural': {'result': 'success'}}), {})

    def test_each_non_success_result_blocks_the_gate(self):
        for result in ['failure', 'cancelled', 'skipped', 'unknown']:
            with self.subTest(result=result):
                self.assertEqual(check({'rust': {'result': 'success'}, 'neural': {'result': result}}), {'neural': result})

    def test_absent_or_malformed_results_fail_closed(self):
        for result in [None, {}, []]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                check(result)
        self.assertEqual(check({'neural': {}}), {'neural': 'missing result'})
        self.assertEqual(check({'neural': None}), {'neural': 'missing result'})


if __name__ == '__main__':
    unittest.main()
