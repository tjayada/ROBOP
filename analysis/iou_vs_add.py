#!/usr/bin/env python3
"""Per-frame CNOS/SAM detection-mask IoU (vs ground-truth mask) joined with per-frame ADD.

Reports, per (model, dataset), the Spearman rank correlation between mask IoU
and ADD and the median ADD in the worse / better half of frames by IoU.
Baxter has no ground-truth masks and is excluded.

Note: the gating iou_cache quantity is a DIFFERENT IoU (mask rendered at the
estimated pose vs detection), the confidence-gating signal. Do not mix.

Stdlib only. Panda-Orb decodes 32,315 RLE masks and takes ~15 min.

    python -u analysis/iou_vs_add.py --gt-masks <dir> --detections <dir> \\
        --results <results_root>
"""
import argparse
import gzip
import json
import statistics
import zipfile
from pathlib import Path

DATASETS = [("Panda-Orb", "panda_orb", "gt_panda_orb"), ("OWI", "craves", "gt_craves"),
            ("LBR", "lbr_med7", "gt_hydra_lbr"), ("xArm", "xarm", "gt_hydra_xarm"),
            ("Meca", "meca", "gt_hydra_meca")]

MODELS = {
    "MegaPose coarse": ("megapose_coarse", {
        "Panda-Orb": "panda_orb_megapose_coarse_shards_final", "OWI": "megapose_coarse_craves",
        "LBR": "megapose_coarse_hydra_lbr", "xArm": "megapose_coarse_hydra_xarm",
        "Meca": "megapose_coarse_hydra_meca"}),
    "GigaPose coarse": ("gigapose_coarse", {
        "Panda-Orb": "panda_orb_gigapose_shards_final", "OWI": "gigapose_craves",
        "LBR": "gigapose_hydra_lbr", "xArm": "gigapose_hydra_xarm", "Meca": "gigapose_hydra_meca"}),
    "FoundPose coarse": ("foundpose_coarse", {
        "Panda-Orb": "panda_orb_foundpose_shards_final", "OWI": "foundpose_craves",
        "LBR": "foundpose_hydra_lbr", "xArm": "foundpose_hydra_xarm", "Meca": "foundpose_hydra_meca"}),
    "NeMO coarse": ("nemo_coarse", {
        "Panda-Orb": "panda_orb_nemo_shards_final", "OWI": "nemo_craves",
        "LBR": "nemo_hydra_lbr", "xArm": "nemo_hydra_xarm", "Meca": "nemo_hydra_meca"}),
    "FoundPose refined": ("foundpose_coarse_megepose_refiner", {
        "Panda-Orb": "panda_orb_refine_foundpose_shards_final", "OWI": "megapose_refiner_foundpose_craves",
        "LBR": "megapose_refiner_foundpose_hydra_lbr", "xArm": "megapose_refiner_foundpose_hydra_xarm",
        "Meca": "megapose_refiner_foundpose_hydra_meca"}),
    "GigaPose refined": ("gigapose_coarse_megapose_refiner", {
        "Panda-Orb": "panda_orb_gigapose_megapose_shards_final", "OWI": "gigapose_megapose_craves",
        "LBR": "gigapose_megapose_hydra_lbr", "xArm": "gigapose_megapose_hydra_xarm",
        "Meca": "gigapose_megapose_hydra_meca"}),
}


def load_gz(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def load_results(d):
    if (d / "results.json").exists():
        return json.load(open(d / "results.json"))
    z = d / "results.json.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            return json.loads(zf.read([n for n in zf.namelist() if n.endswith(".json")][0]))
    return None


def rle_to_int(seg):
    counts, size = seg["counts"], seg["size"]
    buf = bytearray(size[0] * size[1])
    pos = 0
    for i, run in enumerate(counts):
        if i % 2 == 1:
            buf[pos:pos + run] = b"\x01" * run
        pos += run
    return int.from_bytes(buf, "little")


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-masks", type=Path, required=True, help="dir of gt_<ds>.json.gz")
    ap.add_argument("--detections", type=Path, required=True, help="dir of <ds>.json.gz (CNOS/SAM)")
    ap.add_argument("--results", type=Path, required=True, help="results root")
    args = ap.parse_args()
    GT, DET, RES = args.gt_masks, args.detections, args.results

    IOU = {}
    for label, det_stem, gt_stem in DATASETS:
        gt = {d["frame_id"]: d for d in load_gz(GT / f"{gt_stem}.json.gz")["detections"]}
        sam = {d["frame_id"]: d for d in load_gz(DET / f"{det_stem}.json.gz")["detections"]}
        m = {}
        for fid, g in gt.items():
            s = sam.get(fid)
            if s is None or not s.get("found") or not g.get("found"):
                continue
            a, b = rle_to_int(g["segmentation"]), rle_to_int(s["segmentation"])
            inter, union = bin(a & b).count("1"), bin(a | b).count("1")
            ga, sa = bin(a).count("1"), bin(b).count("1")
            m[fid] = (inter / union if union else 0.0, inter / ga if ga else 0.0, inter / sa if sa else 0.0)
        IOU[label] = m
        v = sorted(m.values())
        print(f"{label:10s} n={len(m):6d}  IoU mean={statistics.fmean(x[0] for x in v):.3f} "
              f"p10={sorted(x[0] for x in v)[len(v)//10]:.3f}  |  recall mean={statistics.fmean(x[1] for x in v):.3f} "
              f"p10={sorted(x[1] for x in v)[len(v)//10]:.3f}  |  "
              f"precision mean={statistics.fmean(x[2] for x in v):.3f}", flush=True)

    print("\n%-19s %-10s %6s %8s %10s %10s %8s"
          % ("model", "dataset", "n", "rho", "medADD_lo", "medADD_hi", "ratio"))
    for mname, (root, dmap) in MODELS.items():
        for label, _, _ in DATASETS:
            R = load_results(RES / root / dmap[label])
            if R is None:
                print(f"  {mname}/{label}: MISSING", flush=True)
                continue
            m = IOU[label]
            pairs = [(m[q["frame_id"]][0], q["add_m_mm"]) for q in R["queries"]
                     if q.get("frame_id") in m and q.get("add_m_mm") is not None]
            if len(pairs) < 10:
                print(f"  {mname}/{label}: n={len(pairs)}", flush=True)
                continue
            i_, a_ = zip(*pairs)
            med = statistics.median(i_)
            lo = [a for i, a in pairs if i < med]
            hi = [a for i, a in pairs if i >= med]
            ml, mh = statistics.median(lo), statistics.median(hi)
            print("%-19s %-10s %6d %+8.3f %10.1f %10.1f %8.2f"
                  % (mname, label, len(pairs), spearman(list(i_), list(a_)), ml, mh,
                     ml / mh if mh else float("nan")), flush=True)


if __name__ == "__main__":
    main()
