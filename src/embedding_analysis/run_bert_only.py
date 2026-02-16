"""Run embedding analysis scripts (excluding GCN for now)."""

from pathlib import Path
import subprocess
import sys

SCRIPTS = [
    "analyze_synthetic_embeddings.py",
    "analyze_real_embeddings.py",
    "compare_embeddings.py",
    "analyze_user_embeddings.py",
    "analyze_nearest_neighbors.py",
]


def run_script(script_path: Path) -> bool:
    """Run a single BERT analysis script and return True on success."""
    print(f"\n{'=' * 60}")
    print(f"Running: {script_path.name}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent,
        capture_output=False,
    )
    return result.returncode == 0


def main():
    """Run the subset of embedding analysis scripts that use BERT embeddings only."""
    script_dir = Path(__file__).parent

    print("=" * 60)
    print("EMBEDDING ANALYSIS - BERT ONLY (NO GCN)")
    print("=" * 60)

    results = {}
    for script in SCRIPTS:
        script_path = script_dir / script
        if script_path.exists():
            success = run_script(script_path)
            results[script] = "SUCCESS" if success else "FAILED"
        else:
            print(f"\nScript not found: {script}")
            results[script] = "NOT_FOUND"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for script, status in results.items():
        print(f"  {script}: {status}")

    success_count = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nCompleted: {success_count}/{len(SCRIPTS)} scripts")


if __name__ == "__main__":
    main()
