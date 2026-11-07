# Layout and Git provenance

The initial commit `8ceb2a1` imported the participant package with data excluded. Every tracked file from that commit is preserved byte for byte under `original-challenge/`, including the original `.gitignore`, package metadata, starter results and documentation. The initial package did not contain a root README.md. No later experiment files were included in this snapshot.

`ae14121` added the evaluation harness and baseline reproduction. These belong to `baseline-reproduction/`. Subsequent commits `e39105d`–`aab874a` added and evolved the B4 detector, its helper scripts, causality test, experiment log and results; these belong to `physics-context-detector/`. B4's version numbers refer to successive revisions of one detector, not independent current implementations. Historical variants remain available through Git history.

The B1 evaluator is copied into B4 for independent execution and local output ownership. Both import the original scorer; neither changes the reference package. B4 was moved out of the original Python package and imports its physics module explicitly. Script paths now resolve shared input data relative to the repository and output directories relative to each experiment.

Shared documents belong here in `docs/`. Original challenge documents remain in the reference snapshot, and experiment-specific notes stay with their experiment. Existing shared datasets and the local virtualenv are retained; relocated duplicate source trees and their generated caches were removed. No dataset files were deleted. The root README replaces the previous exclusion-only note (commit `c6cc266`); its data and generated-stress exclusions remain documented.

To run original challenge tests against shared data, set `RETROFIT_DATA_FULL` and `RETROFIT_DATA_STARTER` to the absolute shared dataset paths. Original instructions and defaults inside the reference snapshot intentionally retain their original contents.

Windows PowerShell, from the repository root after the [Windows setup](../README.md#windows-powershell):

```powershell
$env:RETROFIT_DATA_FULL = (Resolve-Path .\data\full).Path
$env:RETROFIT_DATA_STARTER = (Resolve-Path .\data\starter).Path
.\.venv\Scripts\python.exe -m pytest -q .\original-challenge\tests
```

These environment variables apply to the current PowerShell session. The shared datasets must be present; see [data setup](../data/README.md).

## Reorganization verification

All 83 original tracked files were compared byte for byte with `8ceb2a1`. All experiment Python files passed syntax parsing. The relocated B1 evaluator reproduced train 58.2, validation 78.0, mean 68.1 from the saved predictions. The B4 CLI loaded successfully and rejected an output path under shared `data/` before creating it. Full B4 execution and pytest could not be completed in the current environment: the local virtualenv lacks dependencies, the system Python lacks a Parquet engine and pytest, and dependency download was unavailable.
