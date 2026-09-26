import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

test_files = [
    "test_phase1.py",
    "test_phase2.py",
    "test_phase3.py",
    "test_phase4.py",
    "test_phase5.py",
    "test_phase6.py",
    "test_phase7.py",
    "test_phase8.py",
    "test_phase9.py",
    "test_phase10.py",
    "test_phase11_e2e.py",
    "test_gemini_hardening.py",
    "test_phase12_auto_editor.py",
    "test_vieneu_real.py",
    "test_gemini_status_ui.py",
    "test_phase12_batch_auto_editor.py"
]

def run_regression():
    print("==========================================================")
    print("     RUNNING FULL MASTER REGRESSION SUITE (PHASES 1-12+)  ")
    print("==========================================================")

    results = {}
    for tf in test_files:
        print(f"\n>>> Running {tf} ...")
        res = subprocess.run([sys.executable, tf], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            print(f"    [PASS] {tf}")
            results[tf] = "PASS"
        else:
            print(f"    [FAIL] {tf}")
            print("--- STDOUT ---")
            print(res.stdout[-500:])
            print("--- STDERR ---")
            print(res.stderr[-500:])
            results[tf] = "FAIL"

    print("\n==========================================================")
    print("              REGRESSION TEST SUMMARY                     ")
    print("==========================================================")
    passed_count = sum(1 for v in results.values() if v == "PASS")
    total_count = len(results)

    for tf, status in results.items():
        print(f"  {tf:<22} : {status}")

    print(f"\nTotal: {passed_count}/{total_count} PASSED")
    if passed_count == total_count:
        print("ALL TESTS PASSED! ZERO REGRESSIONS.")
        return 0
    else:
        print("SOME TESTS FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(run_regression())
