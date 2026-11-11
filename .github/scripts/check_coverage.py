"""Fail when coverage drops below the project's thresholds.

Run after pytest with coverage:
    coverage json -o coverage.json
    python .github/scripts/check_coverage.py coverage.json
Thresholds: 80 % overall, 90 % for each module of the package.
"""

import json
import sys
from pathlib import Path

OVERALL = 80.0
PER_MODULE = 90.0


def main(path: str = "coverage.json") -> int:
    """Print per-module coverage and return 1 if a threshold is missed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    total = data["totals"]["percent_covered"]
    problems = []
    if total < OVERALL:
        problems.append(f"overall coverage {total:.1f} % is below {OVERALL:.0f} %")
    for name, info in sorted(data["files"].items()):
        module = name.replace("\\", "/")
        if "/topocorr/" not in module:
            continue
        percent = info["summary"]["percent_covered"]
        status = "ok" if percent >= PER_MODULE else "LOW"
        print(f"{status:4s} {percent:6.1f} %  {module.split('/topocorr/')[-1]}")
        if percent < PER_MODULE:
            problems.append(f"{module} coverage {percent:.1f} % is below {PER_MODULE:.0f} %")
    print(f"overall {total:.1f} %")
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
