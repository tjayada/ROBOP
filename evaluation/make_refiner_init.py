"""Drop failed pose estimates from a results.json for refiners that cannot
consume them.

Writes <stem>_for_refiner.json next to each input. Failed frames
(pnp_failed=true, est_pose=null) are removed, everything else is kept as-is;
when scoring the refined output, add the dropped frames back with the 1000 mm
failure sentinel (run_eval_panda_orb.py) so metrics stay comparable to a
refiner that propagates failures.

Note: ROBOP's own megapose_refiner needs the FULL file - it pairs frames by
frame_key and raises on any frame missing from the init JSON
(megapose_refiner_estimator._InitPoseStore).

    python evaluation/make_refiner_init.py results.json [more.json ...]
"""

import argparse
import json
from pathlib import Path


def make_refiner_init(path: Path) -> Path:
    results = json.loads(path.read_text())
    queries = results["queries"]
    kept = [q for q in queries if not q.get("pnp_failed")]
    results["queries"] = kept

    out = path.with_name(path.stem + "_for_refiner.json")
    out.write_text(json.dumps(results))
    print(f"{out}: {len(queries)} -> {len(kept)} queries "
          f"({len(queries) - len(kept)} failed dropped)")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Write <stem>_for_refiner.json with failed queries dropped.")
    ap.add_argument("results", nargs="+", type=Path, help="results.json files to filter.")
    args = ap.parse_args()
    for path in args.results:
        make_refiner_init(path)


if __name__ == "__main__":
    main()
