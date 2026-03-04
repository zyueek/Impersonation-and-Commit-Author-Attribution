# Commit-Level Style Imitation Attack: Detailed Technical README

This document explains, in implementation-level detail, how the attack in `adversial/` is conducted, and exactly how it adapts and changes the approach from `adv.pdf` (Quiring et al., USENIX Security 2019) to a **commit-snippet** setting.

It is intended to complement `adversial/README.md` and make the adaptation choices explicit.

## 1) What Problem This Attack Solves

The input in this repository is usually a commit-level JSON item:

- `added_code`: list of added lines (or string)
- author id field: `emailname` or `author`
- optional metadata: `message`, `filename`, `language`

The attacker rewrites `added_code` so that a snippet from true author `A` looks stylistically closer to target author `B`, while keeping content plausible and mostly semantics-preserving.

The implemented mode is **targeted imitation**.

## 2) High-Level Mapping to `adv.pdf`

`adv.pdf` introduces:

- black-box authorship attack
- semantics-preserving source-to-source transformations
- rich transformation families (control/declaration/API/template/misc)
- search-guided sequence construction using Monte-Carlo Tree Search (MCTS)
- compiler-aware infrastructure (Clang AST/CFG, def-use/declaration mappings)

This repository keeps the same high-level adversarial goal (targeted style imitation), but changes the mechanics to fit commit snippets:

- operate on diff text (`added_code`) instead of full compilable artifacts
- prefer lightweight deterministic rewrite passes over search
- AST/CFG/UDC is available only for parseable Python snippets
- non-Python uses token/text heuristics with conservative guards
- no compile/test oracle; semantics preservation is best-effort

## 3) End-to-End Attack Pipeline in This Repository

Primary scripts:

- Single JSON attack generation: `adversial/attack_dataset.py`
- Batch attack generation: `adversial/run_attack_all.py`
- Core rewrite engine: `adversial/imitate_style.py`
- Style profiling: `adversial/style_profile.py`
- Python AST attack layer: `adversial/ast_transforms.py`
- Non-Python token-aware renaming: `adversial/token_transforms.py`

### Step 0: Parse Labels and Author IDs

From each input item:

- label field for evaluation task is detected (`emailname` / `author` / `label`)
- author field for profile construction is detected (`emailname` or `author`)
- code is normalized by joining `added_code` lines into text

Code references:

- `adversial/load_data.py`
- `adversial/attack_dataset.py`

### Step 1: Build Target Author Style Profiles

For each author, we aggregate that author's code snippets and infer a `StyleProfile`:

- indentation kind: tabs vs spaces
- indentation size (if spaces): heuristic preference for 2 or 4, else median-like fallback
- brace style: `k&r` vs `allman`
- keyword spacing: `if (` vs `if(`
- quote preference: `'` vs `"`
- identifier style: `snake` vs `camel` vs `other`
- loop preference: `for` vs `while` vs `none`
- identifier pool: top frequent identifiers (for stronger renaming)

Code reference:

- `adversial/style_profile.py`

Important implementation detail:

- `attack_dataset.py` builds profiles from the whole given JSON.
- `eval_imitation.py` builds profiles from the **training split** only (to avoid obvious leakage in evaluation).

### Step 2: Choose Target Author per Sample

For each attacked sample:

- true author is identified from task-specific label/author field
- target author is selected:
  - random `B != A`, or
  - `--fixed_target`, with fallback to a different author when needed

Attack coverage can be controlled by:

- `--fraction` in `attack_dataset.py` / `run_attack_all.py`

### Step 3: Rewrite the Snippet (Layered Attack)

The rewrite pipeline in `targeted_attack(...)` is:

1. optional Python AST attack (`--use_ast`)
2. optional token-aware renaming for Go/Java/JS/PHP (`--token_rename_max > 0`)
3. text-level style normalization (all languages)

Code reference:

- `adversial/imitate_style.py`

#### 3A) Python AST/CFG/UDC Layer (when parseable)

Implemented in `adversial/ast_transforms.py`.

Transforms:

- `for -> while` rewrite (`for_to_while`)
  - applied only when target profile prefers `while`
  - rewrites loop using `iter(...)`, `next(...)`, `StopIteration`
- CFG-guided no-op insertion (`cfg_insert_noop`)
  - builds a lightweight statement-level CFG approximation
  - inserts `pass` in a simple safe block with single successor
- UDC-guided local renaming (`rename_locals_with_udc`)
  - computes defs/uses from Python AST `Name` contexts
  - renames consistently to target identifier style (`snake`/`camel`)
  - avoids unsafe identifiers and collisions

The AST layer returns an `AstAttackReport` with:

- `applied`: list of applied transforms
- `rename_map`: local identifier mapping

Those are propagated to dataset output as `attack_meta`.

#### 3B) Token-Aware Identifier Renaming for Non-Python

Implemented in `adversial/token_transforms.py`, enabled by `--token_rename_max`.

Language support:

- `go`, `java`, `js`, `php`

Mechanism:

- mask/skip spans for comments and string literals
- collect declared identifiers by language-specific regex heuristics
- generate mapping toward target style + optional target `ident_pool` names
- avoid reserved/problematic names (language-specific lists)
- apply renaming outside skip spans and avoid obvious member-access/key contexts

Output metadata:

- `token_rename_declared`
- `token_rename_map`

#### 3C) Text-Level Style Normalization (All Languages)

Implemented in `adversial/imitate_style.py`.

Transforms:

- indentation conversion (tabs/spaces and width adaptation)
- brace style normalization (`allman` vs `k&r`)
- keyword spacing normalization (`if (` vs `if(` etc.)
- conservative quote conversion where safe
- lightweight identifier renaming from simple assignment/declaration patterns

This layer is the guaranteed fallback when AST/token-aware layers do not apply.

### Step 4: Write Attacked Dataset

For each attacked item:

- `added_code` is replaced with rewritten lines
- `attack_target` is recorded
- `attack_meta` is optionally added when AST/token-aware edits were applied

Script:

- `adversial/attack_dataset.py`

### Step 5: Batch Attack Over All Repositories

`adversial/run_attack_all.py` iterates over `language/combined_{lang}/*.json`, invokes `attack_dataset` for each file, and writes outputs mirroring folder structure under `--out_root`.

Useful options:

- `--langs all` or comma list
- `--use_ast/--no-use_ast`
- `--token_rename_max N` (stronger non-Python imitation)
- `--fraction`
- `--manifest` JSONL logging

## 4) Evaluation Protocol (How Degradation Is Measured)

Main scripts:

- single dataset: `adversial/eval_imitation.py`
- batch across repos: `adversial/run_eval_all_attacked.py`

Protocol:

1. split clean data into train/test
2. train model on clean train
3. evaluate clean test
4. evaluate attacked version of the same test items
5. report metric drops (`clean - attacked`) and targeted success rate

Supported models include:

- `rf` (weak / weak_info / strong alias)
- `gcb` (embed or finetune)
- `lang`, `t5`
- `gcb_info`, `lang_info`, `t5_info`

The `*_info` variants include message/filename channels, which are intentionally unchanged by this attack.

## 5) Exact Adaptations and Changes from `adv.pdf` to Commit Setting

This section is the key "what changed" summary.

### Change A: Attack Unit

From `adv.pdf`:

- full source artifacts/challenge files (compiler-friendly units)

To this repository:

- commit snippets (`added_code` hunks), often partial and context-limited

Reason:

- repository data is commit-level JSON, not full buildable projects.

### Change B: Transformation Engine

From `adv.pdf`:

- 36 source-to-source transformations over rich compiler IR support

To this repository:

- compact transformation set focused on local edits:
  - Python AST/CFG/UDC subset when parseable
  - token-aware renaming for selected non-Python languages
  - text/style normalization fallback for all languages

Reason:

- many commit hunks are not parseable/compilable standalone, so global refactors are unsafe.

### Change C: Search Strategy

From `adv.pdf`:

- iterative black-box search with MCTS guided by model scores

To this repository:

- deterministic profile-driven rewrite passes (no MCTS loop, no classifier-query search)

Reason:

- prioritize reproducibility and speed for large batch commit evaluation.
- commit snippets provide limited transformation freedom; expensive search often has low marginal gain.

### Change D: Semantics Assurance

From `adv.pdf`:

- compiler tooling and transformation validation in a full-code context

To this repository:

- best-effort semantic safety via conservative local rewrites
- no compile/test oracle in attack pipeline

Reason:

- standalone commit hunks usually cannot be compiled/tested in isolation.

### Change E: Feature Scope

From `adv.pdf`:

- primary stylometric manipulation in code artifact context

To this repository:

- code-only attack on `added_code`
- commit message/path are left untouched

Reason:

- editing message/path would change commit intent or affected files, reducing realism of commit-time impersonation.
- this also explains why multi-modal `*_info` models are often more robust.

### Change F: Layout Policy

`adv.pdf` discusses a stricter setting that avoids layout changes to make the attack harder.

This repository includes layout-alignment transforms (indent/brace/spacing) because:

- commit snippets often expose strong formatting cues
- these edits are local and plausible in diffs
- they provide coverage when deeper AST transforms are unavailable

### Change G: Commit-Plausibility Constraint

Adaptation introduced here:

- keep edits local to snippet
- avoid large structural refactors that would explode diff size or require cross-file context
- preserve the visual/operational nature of a "real commit hunk"

This is not just a tooling limitation; it is a deliberate threat-model choice.

## 6) Why These Changes Matter for Reported Results

Compared to `adv.pdf`, measured average drops can be smaller in commit evaluation because:

- attack coverage is lower on partial/non-parseable snippets
- metadata channels (message/path) are untouched for multi-modal models
- per-repo test sets are often small (metric quantization, many zeros)
- targeted imitation from sparse per-author commit evidence is harder than from full-code corpora

For details and paper-ready discussion:

- `adversial/WHY_DROPS_SMALL.md`

## 7) Reproducible Commands

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
  --task authorship \
  --seed 42 \
  --test_size 0.2 \
  --gcb_mode embed \
  --gcb_local_files_only
```

## 8) Practical Interpretation

This implementation should be interpreted as a **commit-level targeted style imitation simulator** inspired by `adv.pdf`, not a full reproduction of its compiler-driven MCTS attack.

The main contribution of this adaptation is practical compatibility with large, real commit datasets while still preserving the core adversarial question:

"If an attacker rewrites only commit code style toward a target developer, how much does authorship detection degrade?"
