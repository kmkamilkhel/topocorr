"""Fail when coverage drops below the project's thresholds.

Run after pytest with coverage:
    coverage json -o coverage.json
    python .github/scripts/check_coverage.py coverage.json [--offline]
Thresholds: 80 % overall, 90 % for each module of the package.

With --offline (test jobs without Earth Engine credentials), the modules that build Earth Engine
computations are reported but not checked: their code runs only in the Earth Engine tests, which
check every module.
"""

import json
import sys
from pathlib import Path

OVERALL = 80.0
PER_MODULE = 90.0
EARTH_ENGINE_MODULES = {
    "artefact.py",
    "correction.py",
    "illumination.py",
    "masking.py",
    "model.py",
    "sensors.py",
    "terrain.py",
}


def main(path: str = "coverage.json", *flags: str) -> int:
    """Print per-module coverage and return 1 if a threshold is missed."""
    offline = "--offline" in flags
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    covered = statements = 0
    for name, info in sorted(data["files"].items()):
        module = name.replace("\\", "/")
        if "/topocorr/" not in module:
            continue
        short = module.split("/topocorr/")[-1]
        percent = info["summary"]["percent_covered"]
        if offline and short in EARTH_ENGINE_MODULES:
            print(f"ee   {percent:6.1f} %  {short} (checked in the Earth Engine tests)")
            continue
        covered += info["summary"]["covered_lines"]
        statements += info["summary"]["num_statements"]
        status = "ok" if percent >= PER_MODULE else "LOW"
        print(f"{status:4s} {percent:6.1f} %  {short}")
        if percent < PER_MODULE:
            problems.append(f"{module} coverage {percent:.1f} % is below {PER_MODULE:.0f} %")
    total = 100.0 * covered / statements if statements else 100.0
    if total < OVERALL:
        problems.append(f"overall coverage {total:.1f} % is below {OVERALL:.0f} %")
    print(f"overall {total:.1f} %" + (" (offline modules)" if offline else ""))
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
