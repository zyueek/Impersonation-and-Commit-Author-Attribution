# Adversarial Commit Imitation (Core Files)

This folder contains the main files used to run commit-level targeted style imitation attacks and evaluate attribution degradation.

## Key Scripts

- `attack_dataset.py`: generate attacked samples for one dataset file.
- `run_attack_all.py`: batch attack generation across repositories/languages.
- `imitate_style.py`: layered rewrite engine (AST/token/text style changes).
- `style_profile.py`: per-author style profile construction.
- `ast_transforms.py`: Python AST/CFG/UDC-based transformations.
- `token_transforms.py`: token-aware identifier renaming for non-Python languages.
- `eval_imitation.py`: evaluate clean vs attacked performance.
- `run_eval_all_attacked.py`: batch evaluation driver.
- `load_data.py`, `common.py`: shared data loading and utilities.

## Method Documentation

- `README_COMMIT_ATTACK_DETAILS.md` explains how the attack pipeline is adapted to commit snippets and how evaluation is performed.

## Typical Usage

Generate attacked datasets:

```bash
python -m adversial.run_attack_all \
  --langs all \
  --out_root adversial/attacked_language_strong \
  --seed 42 \
  --fraction 1.0 \
  --use_ast \
  --token_rename_max 30
```

Evaluate clean vs attacked:

```bash
python -m adversial.run_eval_all_attacked \
  --langs all \
  --language_dir language \
  --attacked_root adversial/attacked_language_strong \
  --out_root adversial/eval_attacked_all_strong_more_models \
  --models rf,gcb,lang,t5,gcb_info,lang_info,t5_info \
  --rf_mode weak \
  --task authorship
```
