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

## Full Data Release (Code Authorship Prediction)

The full release for code authorship prediction repository data is hosted on Zenodo because it is too large to upload to this GitHub repository:

[Zenodo Full Release](https://zenodo.org/records/18777766?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZkZmJiYTQxLTRmNzMtNGE5NC05NzI5LTM1YzZkYjVjMjhhZCIsImRhdGEiOnt9LCJyYW5kb20iOiI0YjNhMDY3OWU0MjY4NTYxYTY5MzBlY2Y0NjdjYTM1YiJ9.uoHirI8s6_zw0t8gUPdtZOvpiG2mRJaQP2UWhakf3ZZelp5xyUVveSmbwaGjPlP2ap4NMwYJqHyXvB8p2ZAUew)

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
