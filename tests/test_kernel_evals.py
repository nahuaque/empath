import unittest

from empath.evals import format_eval_failures, kernel_eval_cases, run_kernel_evals


class KernelEvalSuiteTests(unittest.TestCase):
    def test_eval_catalog_matches_first_pass_scope(self):
        cases = kernel_eval_cases()

        self.assertGreaterEqual(len(cases), 30)
        self.assertLessEqual(len(cases), 50)
        self.assertEqual(len(cases), len({case.name for case in cases}))
        for case in cases:
            self.assertEqual(case.name, case.state.state_id)
            self.assertTrue(
                case.expected_hypotheses
                or case.forbidden_hypotheses
                or case.unsafe_interventions
                or case.acceptable_top
            )

    def test_kernel_eval_suite_passes(self):
        results = run_kernel_evals()
        failures = tuple(result for result in results if not result.passed)

        if failures:
            self.fail(format_eval_failures(failures))


if __name__ == "__main__":
    unittest.main()
