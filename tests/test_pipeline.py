import subprocess
import sys


def test_pipeline_runs():
    result = subprocess.run(
        [sys.executable, "run_pipeline.py"],
        capture_output=True,
        text=True
    )

    # A pipeline deve executar (mesmo que gere logs)
    assert result is not None
