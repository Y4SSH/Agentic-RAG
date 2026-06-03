"""
Simple smoke-test runner for the Agentic RAG project.
Run from the repo root inside the project's venv:

& .venv\Scripts\python.exe scripts\run_smoke_tests.py

Exit codes:
 0 - all tests passed
 1 - test failures
"""

import os
import sys

# Ensure repo root is on sys.path when running this script directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    from src.agent.graph import run_agent

    TESTS_PASSED = True

    print("Running smoke tests...")

    # Test 1: empty docs behavior (expect deterministic empty_response)
    q1 = "asdkfjhasdfkljhasdf"  # unlikely to match any doc
    res1 = run_agent(q1)
    print("Test 1: query ->", q1)
    print("  documents returned:", len(res1.get("documents", [])))
    if len(res1.get("documents", [])) == 0:
        gen = (res1.get("generation") or "").lower()
        if (
            ("sorry" in gen)
            or ("do not contain any information" in gen)
            or ("no information" in gen)
        ):
            print("  -> Passed (empty_response detected)")
        else:
            print("  -> Failed (no empty_response pattern)")
            TESTS_PASSED = False
    else:
        print("  -> Failed (expected 0 documents)")
        TESTS_PASSED = False

    # Test 2: basic grounded response behavior
    q2 = "What is this project about?"
    res2 = run_agent(q2)
    print("\nTest 2: query ->", q2)
    print("  documents returned:", len(res2.get("documents", [])))
    steps = "\n".join(res2.get("steps_taken", []))
    print("  steps_taken:\n", steps)
    if "Hallucination check passed" in steps or "Hallucination check passed" in steps:
        print("  -> Passed (hallucination check passed)")
    else:
        print("  -> Warning: hallucination check not observed; review steps")

    if TESTS_PASSED:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
