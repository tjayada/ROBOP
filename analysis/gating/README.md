# Confidence gating (MaskVal)

Silhouette-IoU gate over the whole detect-then-estimate pipeline (Quentin &
Goehring, arXiv:2409.03556). Run in order:

1. **`render_masks.py`** renders the estimator's silhouette masks
   (`est_mesh_projection`) from an eval results JSON into `est_masks.json.gz`
   (column-major RLE, robot-detector detection schema). Needs the PyTorch3D/GPU stack.
2. **`risk_coverage_from_masks.py`** computes IoU(est mask, CNOS mask) per frame ->
   `figures/gating/iou_cache/` + risk-coverage figures (retained-set ADD-AUC vs
   coverage). GPU-free once the masks exist.
3. **`maskval_operating_points.py`** reads `iou_cache/` -> operating-point tables
   (`gating_tables_ap99.md` main, `gating_tables_ap95.md` appendix) + the MaskVal
   Table III signal-quality summary.

Outputs land in `figures/gating/` under the repo root. Shared helpers:
`../analysis_utils.py`.
