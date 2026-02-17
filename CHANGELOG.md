# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automatic Jupyter kernel detection from virtual environment in `museek_run_notebook`
- Kernel detection searches both visible kernels and venv's share directory
- Helpful error messages with instructions when no kernel is found for a venv
- `merge_slurm_options()` utility function to handle SLURM option overrides (user options override defaults)
- `build_sbatch_script()` utility function to generalize sbatch script generation across CLI commands
- New `museek/cli/common.py` module with reusable CLI option decorators for consistent CLI interfaces
- `--data-path` option in `museek_run_notebook` to specify base data path for notebooks
- `--output-dir` option in `museek_run_notebook` for explicit output directory control

### Changed
- `--kernel` option in `museek_run_notebook` is now optional (no default value)
- When `--kernel` is not specified, the CLI auto-detects the kernel associated with the provided venv
- SLURM options specified by users now properly override default options instead of duplicating them
- Refactored `run_notebook.py` and `run_process_uhf_band.py` to use shared `build_sbatch_script()` function
- Reduced code duplication in sbatch script generation logic
- Split `--output-path` parameter in `museek_run_notebook` into separate `--data-path` and `--output-dir` parameters for improved clarity
- Changed notebook search order in `museek_run_notebook`: absolute path first, then installed package, then current working directory
- SLURM log filenames now standardized with job number: `slurm-<command>-<block_name>-%j.out`
- Job names now include block name: `MuSEEK-Notebook-<block_name>` and `MuSEEK-Process-UHF-<block_name>`
- Improved CLI help text and examples for both `museek_run_notebook` and `museek_run_process_uhf_band`
- Both CLI commands now use shared option decorators from `common.py` for consistency
- Enhanced error messages when notebooks or kernels are not found
- `--venv` option added to both CLI commands for specifying virtual environment path

### Fixed
- Fixed issue where custom SLURM options would be appended rather than overriding defaults
- Fixed dry-run mode to show actual auto-detected kernel name instead of placeholder
- Fixed issue where block name and box number are not passed to papermill
- Fixed path handling in post-calibration notebook
- Fixed path handling to properly support absolute paths, installed packages, and relative paths

## [0.4.1] - 2026-02-03

### Added
- Notebook templates are now included as package data under `museek/notebooks/` so templates are available after installing MuSEEK (including `pip install git+https://github.com/meerklass/museek.git`).

### Changed
- `museek_run_notebook` now resolves notebooks from the installed package location (`museek/notebooks/`) and from a local `./notebooks/` folder when running from the repository. It also accepts absolute paths to notebook files.
- Path handling in the notebook CLI was modernized to use `pathlib.Path`.
- Documentation updated to reflect packaged notebooks and absolute-path support.

### Fixed
- Fixed a bug where absolute notebook paths produced malformed output filenames when generating sbatch scripts; the output filename now uses the notebook stem.
- Improved CLI error messages and dry-run behavior for missing notebooks.


## [0.4.0] - 2026-01-27

### Added
- Pure Python CLI implementation for `museek_run_process_uhf_band` command using Click framework
- Pure Python CLI implementation for `museek_run_notebook` command using Click framework
- Shared SLURM utilities module (`museek/cli/slurm_utils.py`) for sbatch script submission
- Comprehensive help text for main `museek` CLI command
- Auto-display help when `museek` is called without arguments
- Automatic Jupyter kernel validation and installation in `museek_run_notebook`
- `--dry-run` flag for both SLURM submission commands to preview generated sbatch scripts
- GitHub Actions CI workflow for automated testing across Python 3.10-3.12
- Code coverage reporting with pytest-cov
- Pre-commit hooks with ruff for code formatting and linting
- Contributing guidelines and development setup documentation in README
- Pytest configuration with coverage settings

### Changed
- Migrated from argparse to Click for all CLI implementations
- Renamed command from `museek_process_uhf_band` to `museek_run_process_uhf_band` to avoid naming conflicts
- Updated all documentation references from shell script names (`.sh` suffix) to new Python CLI command names
- Enhanced main CLI with detailed pipeline documentation
- Modernized entry points in `pyproject.toml` to use pure Python implementations
- Updated packaging for NumPy ≥2.0.0 compatibility
- Python version constraint updated to >=3.10,<3.13
- Refactored entire codebase with comprehensive type hints and import cleanups
- Made time analysis tests deterministic and timezone-aware
- Switched from flake8/black/isort to ruff for unified linting and formatting
- Updated package metadata and classifiers for Python 3.10-3.12 support

### Removed
- Shell script wrappers (`museek_process_uhf_band.sh`, `museek_run_notebook.sh`) replaced by Python CLIs
- Obsolete `museek/cli/scripts.py` wrapper module
- `.sh` suffix from all command names in documentation

### Fixed
- Shell scripts now work correctly in editable pip installations
- Eliminated code duplication in SLURM sbatch submission logic
- Resolved naming conflict between CLI and config modules
- Fixed circular import issues with TYPE_CHECKING guards
- Corrected microsecond-level assertion tolerances in time-related tests
- Normalized datetimes to UTC for deterministic test behavior across timezones
- Fixed test randomness for reproducible test results

---

## [0.3.3] - 2026-01-13

### Changed
- Updated bash scripts for better functionality ([#181](https://github.com/meerklass/museek/pull/181))
- Pinned ivory to version with numpy<2.0.0 ([#185](https://github.com/meerklass/museek/pull/185))
- Pinned ivory version back to v2.0.0 for Numpy2 support ([#189](https://github.com/meerklass/museek/pull/189))

### Added
- Papermill notebook support ([#166](https://github.com/meerklass/museek/pull/166))

### Fixed
- Build update and dynamic versioning ([#183](https://github.com/meerklass/museek/pull/183))
- Various bugs fixed with Numpy2 update ([#186](https://github.com/meerklass/museek/pull/186))

## [0.2.2] - 2025-10-14

### Fixed
- Import of definitions.py ([#179](https://github.com/meerklass/museek/pull/179))

## [0.2.1] - 2025-10-09

### Changed
- Updated README documentation ([#178](https://github.com/meerklass/museek/pull/178))

## [0.2.0] - 2025-10-06

### Added
- Gain calibration functionality ([#157](https://github.com/meerklass/museek/pull/157))

### Fixed
- Updated requirements.txt to fix dependency issues ([#155](https://github.com/meerklass/museek/pull/155))

### Contributors
- @philbull made their first contribution

## [0.1.0] - 2025-10-06

Initial release with core MuSEEK functionality developed by @amadeuswi, @piyanatk, @wkhu-astro, @spinemart, and @mariogrs.

### Added
- Framework for data processing pipelines using Ivory
- Time-ordered data handling and processing
- RFI mitigation with AOFlagger integration
- Noise diode calibration
- Point source flagging
- Gain calibration plugins
- Standing wave correction and fitting
- Antenna sanity checking
- Zebra pattern removal
- Map making capabilities
- Configuration management and context handling
- CLI interface with `museek` command
- Comprehensive plugin system for data processing steps

### Contributors
- @amadeuswi - Initial framework and core functionality
- @wkhu-astro - Kurtosis computation, sanity checks, raw data flagging
- @spinemart - Straggler identification
- @mariogrs - Median calculation of visibilities
- @piyanatk - Sanity check updates, refactoring, and maintenance
