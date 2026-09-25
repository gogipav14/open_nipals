"""Run tests/test_simca_invariance.py over many data seeds.

The test suite uses one seed per dataset; this sweep is the search that
found most of the edge cases. Usage (from the repository root):

    PYTHONPATH=src JAX_PLATFORMS=cpu python validation/sweep_simca_invariance.py 0 30
"""

import sys

import pytest

sys.path.insert(0, "tests")
import test_simca_invariance as invariance  # noqa: E402

make_data = invariance.make_data
failures = {}


class Collect:
    def __init__(self, seed):
        self.seed = seed

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.failed:
            test_id = report.nodeid.split("::")[1]
            failures.setdefault(test_id, []).append(self.seed)


for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    invariance.make_data = lambda kind, seed=seed: make_data(kind, seed)
    pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:warnings",
            "tests/test_simca_invariance.py",
        ],
        plugins=[Collect(seed)],
    )

for test_id, seeds in sorted(failures.items()):
    print("FAIL", test_id, "seeds", sorted(set(seeds)))
print("DONE", len(failures), "failing test ids")
