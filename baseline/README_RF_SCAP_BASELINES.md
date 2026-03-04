# RF + SCAP Non-PLM Baselines: Method, Adaptation, and Reviewer-Facing Notes

This document explains the two non-PLM baselines in `baseline/`:

- `rf` (Random Forest stylometry baseline; likely what you referred to as "rl")
- `scap` (Source Code Author Profile, byte-level n-gram profile matching)

It also documents how these baselines are adapted from:

- `baseline/sec15-paper-caliskan-islam.pdf` (Caliskan-Islam et al., USENIX Security 2015)
- `baseline/popets-2019-0053.pdf` (Dauber et al., "Git Blame Who?", PoPETs 2019)

and provides text/material to address reviewer requests:

- add/discuss missing authorship-stylometry references
- compare against at least one non-PLM baseline

---

## 1) What Is Implemented Here

### 1.1 RF baseline (non-PLM, discriminative)

Code:

- `baseline/train_rf_baseline.py`
- `baseline/train_rf_baseline_info.py`

Core behavior:

- Input unit: commit-level sample (`added_code`, optional `message`/`filename`)
- Closed-world multi-class author classification
- Stratified train/test split per repository JSON
- RandomForest classifier over stylometric features

Main RF feature options:

- `word_only` (default in this repo's weak setting): token-level TF-IDF only
- `full`: word TF-IDF + char TF-IDF + lightweight layout/statistical features

Important implementation hooks:

- Feature extraction and layout stats: `train_rf_baseline.py:61`
- RF feature union pipeline: `train_rf_baseline.py:189`
- Weak RF parameterization: `train_rf_baseline.py:184` (`n_estimators=30`, `max_features=20`, default `max_depth=5`)
- Info variant metadata text construction: `train_rf_baseline_info.py:98`, `train_rf_baseline_info.py:131`

### 1.2 SCAP baseline (non-PLM, profile/intersection)

Code:

- `baseline/train_scap_baseline.py`
- `baseline/train_scap_baseline_info.py`

Core behavior:

- Build per-author profile from top byte n-grams on training data
- Build sample profile from top byte n-grams on each test sample
- Predict author with highest profile intersection size

Important implementation hooks:

- Byte n-gram extraction: `train_scap_baseline.py:57`
- Per-author profile build: `train_scap_baseline.py:111`
- Intersection-score prediction: `train_scap_baseline.py:120`
- Info variant metadata text construction: `train_scap_baseline_info.py:75`, `train_scap_baseline_info.py:111`

---

## 2) How These Baselines Relate to Prior Work

## 2.1 Adaptation from Caliskan-Islam et al. (USENIX Security 2015)

Paper context (`sec15-paper-caliskan-islam.pdf`):

- Uses a rich Code Stylometry Feature Set (CSFS): lexical + layout + syntactic (AST-derived) features.
- Uses Random Forest classification with feature-selection via information gain.
- Evaluates mostly complete competition submissions.

Adaptation in this repo:

- Keeps the Random Forest stylometric spirit.
- Replaces heavy AST-dependent CSFS with robust text/layout features suitable for commit snippets.
- Uses repository-level stratified splits over commit JSON datasets.
- Adds fragment controls (`fragment_chars`, `fragment_mode`) to avoid trivial attribution from large diffs.

Why adaptation is needed:

- Commit diffs are often short/partial and can be unparsable as standalone programs.
- Multi-language, real-world repository data differs from the controlled complete-file setting.
- A lightweight baseline is needed for broad reproducibility across many repos/languages.

Explicit deviations from USENIX 2015:

- No AST graph extraction/fuzzy parser in these RF scripts.
- No information-gain feature pruning step.
- Smaller, fixed-capacity RF defaults (weak baseline), not the paper's larger forest setup.
- Optional metadata channel (`*_info`) via bucketed message/path representations.

## 2.2 Adaptation from Dauber et al. PoPETs 2019 ("Git Blame Who?")

Paper context (`popets-2019-0053.pdf`):

- Targets small, incomplete, often uncompilable fragments in collaborative settings.
- Starts from Caliskan-style random forest features and modifies pipeline for sparse fragment data.
- Discusses open-world and confidence-threshold handling.
- Reviews byte-level n-gram prior work (Frantzeskou et al., SCAP family) as relevant baselines.

Adaptation in this repo:

- Uses commit-level fragments as the unit of analysis, aligning with the "small/incomplete" challenge.
- Implements an explicit SCAP baseline (byte n-gram profile intersection) for fragment-friendly attribution.
- Provides configurable short-fragment defaults:
  - RF weak default: `fragment_chars=80`
  - SCAP default: `fragment_chars=20`
- Keeps the closed-world protocol for the main benchmark.

Explicit deviations from PoPETs 2019:

- No open-world reject/threshold calibration in these baseline scripts.
- No account-level multi-sample aggregation step in this baseline code path.
- Uses this repo's per-repository commit JSON format rather than git-blame AST feature pipeline.

---

## 3) Baseline Details You Can Cite in Methods

### 3.1 Shared protocol choices (RF and SCAP)

- Author filtering: keep classes with at least `min_samples_per_author` (default 5)
- Split: `train_test_split(..., stratify=author)`
- Per-repo evaluation over `language/combined_{lang}/*.json`
- Confidence output when available (RF via `predict_proba`, SCAP via overlap fraction)

### 3.2 RF weak baseline (code-only, non-PLM)

- Tokenized word TF-IDF over code/text sequence
- Random Forest classifier
- Defaults tuned for weak baseline:
  - 30 trees
  - max_features=20
  - max_depth=5
  - word_max_features=800
  - fragment length 80 chars

### 3.3 RF info baseline (code + commit-info, non-PLM)

- Adds structured metadata tokens:
  - bucketed message length stats
  - filename/path representation and optional path tokens
  - bucketed code size stats
- Still non-PLM and lightweight
- Default remains weak capacity to avoid overfitting and keep comparability

### 3.4 SCAP baseline (code-only, non-PLM)

- Byte n-gram profiles:
  - default `ngram_n=5`
  - author profile top-k: `profile_k=200`
  - sample profile top-k: `sample_k=120`
- Prediction by maximum intersection cardinality between sample and author profiles
- Default fragment length 20 chars (harder attribution setting)

### 3.5 SCAP info baseline (code + commit-info, non-PLM)

- Same SCAP profile/intersection mechanism
- Input text prepended with bucketed message/path/stat tokens
- Useful to quantify how much metadata carries author signal in non-PLM settings

---

## 4) Non-PLM Baseline Comparison (Existing Results in This Repo)

From aggregate files:

- `baseline/aggregate_rf_overall_weak.csv`
- `baseline/aggregate_scap_overall.csv`
- `baseline/aggregate_rf_info_overall_weak.csv`
- `baseline/aggregate_scap_info_overall.csv`

| Baseline | Repos | Accuracy (mean) | Precision_macro (mean) | Recall_macro (mean) | F1_macro (mean) |
|---|---:|---:|---:|---:|---:|
| `rf_weak` | 314 | 0.7226 | 0.5888 | 0.5124 | 0.5075 |
| `scap` | 314 | 0.6425 | 0.5836 | 0.6448 | 0.5649 |
| `rf_weak_info` | 318 | 0.7227 | 0.6009 | 0.5218 | 0.5174 |
| `scap_info` | 318 | 0.7153 | 0.6515 | 0.7054 | 0.6377 |

Interpretation:

- This directly satisfies "compare against at least one non-PLM baseline."
- RF and SCAP behave differently by metric:
  - RF weak has higher mean accuracy.
  - SCAP variants can have stronger macro recall/F1 on these aggregates.
- Adding commit-info benefits both non-PLM families.

---

## 5) Commands to Reproduce Non-PLM Baselines

Run all non-PLM baselines:

```bash
python baseline/run_pipeline.py \
  --lang all \
  --json_dir /home/yueke/author/language \
  --approaches rf,rf_info,scap,scap_info
```

Run only the minimum required non-PLM comparison (RF + SCAP code-only):

```bash
python baseline/run_pipeline.py \
  --lang all \
  --json_dir /home/yueke/author/language \
  --approaches rf,scap
```

---

## 6) Missing References: What to Add and How to Discuss

The reviewer-requested core references should be explicitly cited in the manuscript's related-work + threat-model sections:

1. Dauber et al. (ICSE Companion 2018), "Git Blame Who?"
   - URL: https://dl.acm.org/doi/abs/10.1145/3183440.3195007
   - Why: directly targets small, incomplete fragment attribution in collaborative settings.

2. Matyukhina et al. (CODASPY 2019), "Adversarial authorship attribution in open-source projects"
   - URL: https://doi.org/10.1145/3292006.3300032
   - Why: explicitly studies adversarial style manipulation against attribution.

3. Caliskan-Islam et al. (USENIX Security 2015), "De-anonymizing programmers via code stylometry"
   - URL: https://www.usenix.org/system/files/conference/usenixsecurity15/sec15-paper-caliskan-islam.pdf
   - Why: foundational modern feature+RF stylometry framework.

4. Abuhamad et al. (CCS 2018), "Large-scale and language-oblivious code authorship identification"
   - URL: https://doi.org/10.1145/3243734.3243738
   - Why: large-scale deep-learning baseline family (DL-CAIS line).

5. Dipongkor et al. (2025), "Reassessing Code Authorship Attribution in the Era of Language Models"
   - URL: https://arxiv.org/pdf/2506.17120
   - Why: current PLM/LLM-era reassessment; important context for modern claims.

Additional references to include for stronger coverage:

6. Frantzeskou et al. (2007), "Identifying authorship by byte-level n-grams: the SCAP method"
   - Why: canonical SCAP/source-code profile baseline.

7. Burrows et al. (2007/2009), source-code n-gram attribution lines
   - Why: classical lexical n-gram source-code stylometry baseline family.

8. Quiring et al. (USENIX Security 2019), adversarial attacks on source-code attribution
   - Why: robustness/adversarial context directly relevant to impersonation mitigation claims.

Practical writing guidance:

- Position PLM models as modern strong classifiers, not as "first-ever" authorship methods.
- Frame contribution as: applying/benchmarking modern models in repository-specific commit settings, with robust baselines and adversarial discussion.
- Keep explicit non-PLM baselines (RF/SCAP) in main tables, not only appendix.

Literature-search trace (for response letter transparency):

- Search terms used: `"code authorship attribution"`, `"source code stylometry"`, `"small incomplete source code fragments authorship"`, `"adversarial authorship attribution open source"`, `"SCAP byte-level n-grams source code"`.
- Inclusion criteria: directly relevant to source-code authorship attribution, robustness/adversarial analysis, or baseline feature-model families used in this repo.

---

## 7) Mapping to Reviewer Action Items

Action item: "Add and discuss missing references."

- Addressed by Section 6 above (core + additional references and placement guidance).

Action item: "Compare against at least one non-PLM baseline."

- Addressed by Sections 1-5:
  - implemented non-PLM baselines: RF + SCAP (and info variants)
  - aggregate comparative results already provided
  - reproducible commands provided

---

## 8) Scope and Limitations of These Baselines

- Closed-world classification only in these scripts.
- No explicit open-world reject model (thresholding can be added externally).
- No AST-graph extraction in RF baseline (trade-off for multi-language commit-fragment robustness).
- SCAP uses byte n-grams only; does not model deeper syntax/control structure.

These limitations are acceptable for baseline comparisons but should be explicitly stated in the threats-to-validity section.
