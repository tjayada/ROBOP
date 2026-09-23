# Joint reuse (perturbation sweep + representation bank)

The perturbation sweep measures how the frozen pipeline degrades under joint-angle
error; the representation bank reuses NeMO representations across nearby configs.
`d_surf` (task-space surface displacement, mm) is the shared metric.

## Perturbation sweep

1. **`analyze_perturbation.py`** measures joint-noise robustness on panda-orb: added ADD
   error vs perturbation, the geometry-only floor (the error a rigid re-fit of the
   displaced arm would still incur), and delta* = the reuse tolerance (perturbation
   where the median added error reaches 10% of baseline). Compute from eval sweep
   roots (`scripts/run_joint_noise_sweep.sh`), or `--from-npz <file>` to re-plot
   (numpy + matplotlib only, no geometry deps).

## Representation bank

The expensive artifacts are two 32k eval `results.json`: a **stock** NeMO run and a
**bank** run (`estimator=nemo estimator.type=nemo_bank
+estimator.reuse_bank.tolerance_mm=<delta*>`, which serves the render+encode from a
config-keyed bank when a stored config is within the tolerance). Both store per-frame
`add_m_mm` and, for the bank run, `diagnostics.reuse_bank {hit, d_surf_mm, entry_idx}`.

2. **`analyze_bank.py`** reads the two `results.json` and emits everything, no GPU,
   no re-run: the **figure** (cumulative unique encodes vs frames, the bank
   saturating), the **representation-reuse table** (stock / exact-dedup / bank
   encodes + fractions), the **accuracy table** (ADD-AUC + mean ADD, stock vs bank),
   the **accuracy falsifiers** (aggregate delta vs the run-to-run noise floor,
   hit-frame error vs reuse distance, worst-cluster failure rate), and the
   **cumulative-encode** stat.
3. **`covering_analysis.py`** computes greedy k-center coverage of a joint workload in
   `d_surf`: bank size by tolerance, hit rate, effective dimension.
   `dump_panda_joint_log.py` dumps the panda-orb joint log it reads.
4. **`predict_bank.py`** compares predicted and realized bank economics per dataset.
   It replays the online policy over the stock run's trajectory and prices it with the stock
   per-stage timings.
5. **`compare_bank_accuracy.py`** compares stock and bank accuracy per frame, paired,
   against the tolerance-0 tripwire noise floor.

```
python analysis/joint_reuse/analyze_bank.py <stock_run> <bank_run> --plots figs/bank
```

Shared helpers: `../analysis_utils.py`. `analyze_perturbation.py` carries the
FK/geometry (`RobotGeometry`, `discover_runs`) the other scripts import.
