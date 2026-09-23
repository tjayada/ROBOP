"""IoU of NeMO's OWN bootstrap segmentation mask against the GT arm silhouette.

NeMO stores modal/amodal masks in diagnostics at 224x224 in its crop frame;
bootstrap_scan.crop_params = [x, y, w, h] maps them back to image coordinates.
This is the quantity directly comparable to CNOS/SAM: NeMO is the only estimator
that localises the arm itself, without a detector.

Contrast with est_masks.json.gz from analysis/gating/render_masks.py, whose
header says source = "est_mesh_projection": that is the CAD mesh rendered at
the ESTIMATED POSE, i.e. a pose-quality proxy, not a segmentation output. Pass
--est-masks to report it alongside.

    python analysis/nemo_own_mask_iou.py --gt-masks <dir> --results <nemo_bootstrap_root> \\
        [--est-masks <dir>] [--panda]
"""
import argparse
import gzip
import json
import statistics
from pathlib import Path

DS = [("OWI", "gt_craves", "nemo_bootstrap_craves", "craves"),
      ("LBR", "gt_hydra_lbr", "nemo_bootstrap_hydra_lbr", "hydra_lbr"),
      ("xArm", "gt_hydra_xarm", "nemo_bootstrap_hydra_xarm", "hydra_xarm"),
      ("Meca", "gt_hydra_meca", "nemo_bootstrap_hydra_meca", "hydra_meca")]
DS_PANDA = ("Panda-Orb", "gt_panda_orb", "panda_orb_nemo_bootstrap_shards_final", "panda_orb")


def rle_grid(seg):
    """Uncompressed COCO RLE -> (H, W, bytearray) in column-major flat order."""
    H, W = seg["size"]
    buf = bytearray(H * W)
    pos = 0
    for i, run in enumerate(seg["counts"]):
        if i % 2 == 1:
            buf[pos:pos + run] = b"\x01" * run
        pos += run
    return H, W, buf


def rle_to_int(seg):
    H, W, buf = rle_grid(seg)
    return int.from_bytes(buf, "little")


def squarified_crop(loc, img_hw):
    """Reconstruct the FINAL pass's crop for the 224x224 mask.

    bootstrap_scan.crop_params belongs to the WINNING SCAN PATCH, not to the pass
    that produced modal_mask_rle (see _merge_bootstrap_pass_diagnostics: "masks stay
    the final pass's"). The final crop is the squarified localized bbox, per
    nemo_estimator.py with bbox_crop_pad = 0.0:

        s   = int(min(max(bw, bh), W, H))
        bx0 = clamp(int(cx - s/2), 0, W - s)      # shift-window, not clamp-edge
        by0 = clamp(int(cy - s/2), 0, H - s)

    When the scan localised nothing, the masks are the patch-0 output: square crop
    at the origin.
    """
    IH, IW = img_hw
    if loc is None:
        s = min(IH, IW)
        return 0, 0, s
    x0, y0, x1, y1 = (float(v) for v in loc[:4])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = int(min(max(x1 - x0, y1 - y0), float(IW), float(IH)))
    bx0 = max(0, min(int(cx - s / 2), IW - s))
    by0 = max(0, min(int(cy - s / 2), IH - s))
    return bx0, by0, s


def nemo_mask_to_int(seg224, loc, img_hw):
    """Upsample the 224x224 crop-frame mask into full-image column-major bits."""
    H224, W224, g = rle_grid(seg224)          # column-major: idx = c*H224 + r
    IH, IW = img_hw
    bx0, by0, s = squarified_crop(loc, img_hw)
    out = bytearray(IH * IW)                   # column-major: idx = X*IH + Y
    for X in range(bx0, min(IW, bx0 + s)):
        c = int((X - bx0) * W224 / s)
        if not (0 <= c < W224):
            continue
        base = c * H224
        colbase = X * IH
        for Y in range(by0, min(IH, by0 + s)):
            r = int((Y - by0) * H224 / s)
            if 0 <= r < H224 and g[base + r]:
                out[colbase + Y] = 1
    return int.from_bytes(out, "little")


def _cell(v):
    return f"{statistics.fmean(v):.3f}({len(v)})" if v else "--"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-masks", type=Path, required=True, help="dir of gt_<ds>.json.gz")
    ap.add_argument("--results", type=Path, required=True,
                    help="root holding the nemo_bootstrap_* run dirs (results.json)")
    ap.add_argument("--est-masks", type=Path, default=None, help="dir of <ds>/est_masks.json.gz")
    ap.add_argument("--panda", action="store_true", help="also score Panda-Orb (slow)")
    args = ap.parse_args()
    ds = DS + ([DS_PANDA] if args.panda else [])

    print(f"{'dataset':10s} {'n':>6s} {'modal':>8s} {'amodal':>8s} {'proj':>8s}  (mean IoU vs GT)")
    for label, gt_stem, res_dir, est_stem in ds:
        with gzip.open(args.gt_masks / f"{gt_stem}.json.gz", "rt") as f:
            gtd = {d["frame_id"]: d for d in json.load(f)["detections"]}
        rp = args.results / res_dir / "results.json"
        if not rp.exists():
            print(f"{label:10s} MISSING {rp}")
            continue
        R = json.load(open(rp))
        proj = {}
        if args.est_masks is not None:
            ep = args.est_masks / est_stem / "est_masks.json.gz"
            if ep.exists():
                with gzip.open(ep, "rt") as f:
                    proj = {d["frame_id"]: d for d in json.load(f)["detections"]}
        mo, am, pr = [], [], []
        for q in R["queries"]:
            fid = q["frame_id"]
            g = gtd.get(fid)
            if g is None or not g.get("found"):
                continue
            gi = rle_to_int(g["segmentation"])
            d = q.get("diagnostics", {})
            loc = (d.get("bootstrap_scan") or {}).get("localized_bbox_xyxy")
            ihw = q.get("image_size_hw")
            for key, acc in (("modal_mask_rle", mo), ("amodal_mask_rle", am)):
                seg = d.get(key)
                if seg and ihw:
                    ni = nemo_mask_to_int(seg, loc, ihw)
                    u = bin(gi | ni).count("1")
                    acc.append(bin(gi & ni).count("1") / u if u else 0.0)
            p = proj.get(fid)
            if p and p.get("found"):
                pi = rle_to_int(p["segmentation"])
                u = bin(gi | pi).count("1")
                pr.append(bin(gi & pi).count("1") / u if u else 0.0)
        print(f"{label:10s} {len(R['queries']):6d} {_cell(mo):>8s} {_cell(am):>8s} {_cell(pr):>8s}")


if __name__ == "__main__":
    main()
