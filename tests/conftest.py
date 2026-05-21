import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvq.data import register_all_benchmark_specs  # noqa: E402

register_all_benchmark_specs()
