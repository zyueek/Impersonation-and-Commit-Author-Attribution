# RF + SCAP Baselines (Core Files)

This folder contains the main non-PLM baseline files used for commit-level author attribution evaluation.

## Key Scripts

- `run_pipeline.py`: run selected baseline approaches across languages/repositories.
- `train_rf_baseline.py`: RF baseline (code-only).
- `train_rf_baseline_info.py`: RF baseline with commit metadata features.
- `train_scap_baseline.py`: SCAP baseline (byte n-gram profile intersection).
- `train_scap_baseline_info.py`: SCAP baseline with metadata features.
- `collect_baseline_eval.py`: helper for collecting baseline outputs.
- `collect_rf_info_eval.py`, `collect_scap_eval.py`, `collect_scap_info_eval.py`: metric collection helpers.
- `aggregate_rf_results.py`, `aggregate_rf_info_results.py`, `aggregate_scap_results.py`, `aggregate_scap_info_results.py`: cross-repository aggregation scripts.

 documents method details, adaptation rationale, and reproducible commands.

## Typical Usage

Run code-only RF and SCAP:

```bash
python baseline/run_pipeline.py \
  --lang all \
  --json_dir /home/yueke/author/language \
  --approaches rf,scap
```

Run all RF/SCAP variants:

```bash
python baseline/run_pipeline.py \
  --lang all \
  --json_dir /home/yueke/author/language \
  --approaches rf,rf_info,scap,scap_info
```
