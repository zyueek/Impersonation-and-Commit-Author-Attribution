# Impersonation and Commit Author Attribution

This repository packages the materials for our TSE study on GitHub impersonation risks and commit-level author attribution.

## Repository Contents

### `Quantitative_script`
Scripts used to collect and analyze GitHub data via the GitHub API.

### `Quantitative_data`
Processed outputs from the quantitative pipeline, including repository-level commit and contributor data used in our analysis.
Personal identifiers (for example user IDs, email addresses, and commit SHAs) are anonymized.

### `Interview Script.pdf`
Interview protocol used for participant sessions (except participant S17, who followed a different expert-focused flow).

### `codebook.csv`
Structured coding results for interview responses.

### `Code Attribution`
Main code attribution experiments and model scripts (CodeBERT, GraphCodeBERT, CodeT5, and GPT-based variants), plus summarized results.

### `adversial`
Core commit-level targeted style imitation attack pipeline adapted for partial commit snippets.
This folder includes key scripts for:
- style profiling
- attack generation
- AST/token-based transformations
- clean vs attacked evaluation

See `adversial/README.md` and `adversial/README_COMMIT_ATTACK_DETAILS.md` for technical details.

### `baseline`
Non-PLM baseline implementations and evaluation helpers for RF and SCAP.
`RL` in previous notes refers to the RF stylometry baseline in this repository.

This folder includes:
- RF and RF+info training/evaluation scripts
- SCAP and SCAP+info training/evaluation scripts
- pipeline runners and result aggregation scripts
- aggregate baseline summary CSV files

See `baseline/README.md` and `baseline/README_RF_SCAP_BASELINES.md` for methods and commands.

## Suggested Name on GitHub

`Impersonation and Commit Author Attribution`

## Notes

This is a curated release set. It intentionally includes representative files for adversarial attacks and RF/SCAP baselines rather than every intermediate artifact from the full development workspace.
