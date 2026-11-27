"""Run coordinator tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.test_frame_coordinator import *

if __name__ == "__main__":
    # Run all test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestFrameCoordinatorLifecycle,
        TestHistoryBounds,
        TestComposeEmpty,
        TestSequenceMatching,
        TestQueueProperties,
        TestGenerationValidation,
        TestWorkerResultImmutability,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    sys.exit(0 if result.wasSuccessful() else 1)
