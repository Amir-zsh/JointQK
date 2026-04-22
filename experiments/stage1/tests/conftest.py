import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.stage1.data import register_all_benchmark_specs  # noqa: E402

register_all_benchmark_specs()
