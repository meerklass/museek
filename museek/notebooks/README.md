# MuSEEK Notebook Templates

This directory contains Jupyter notebook templates for MuSEEK data analysis workflows.

## Usage with `museek_run_notebook`

The `museek_run_notebook` CLI tool automatically finds notebooks from the installed package. After installing MuSEEK, simply run:

```bash
museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6
```

### How It Works

Notebooks are included as package data and installed with MuSEEK. The CLI searches for notebooks in this order:

1. **Installed package location** (works for all installation methods)
2. **Current working directory** + `notebooks/`
3. **Absolute path** (if you provide a full path to a notebook file)

### Installation Methods

All of these will work identically - notebooks are always available:

```bash
# Install from GitHub
pip install git+https://github.com/meerklass/museek.git

# Install from local source (non-editable)
pip install /path/to/museek

# Install from local source (editable, for development)
pip install -e /path/to/museek
```

After any of these installations, the notebooks are accessible to `museek_run_notebook`.

### Using Absolute Paths (Optional)

You can also provide an absolute path to any notebook file:

```bash
museek_run_notebook --notebook /path/to/my_custom_notebook.ipynb --block-name 1234 --box 6
```

## Available Notebooks

- `analyze_aoflagger_tracking_results.ipynb` - Analyze AOFlagger tracking results
- `calibrated_data_check_observers.ipynb` - Check calibrated data for observers
- `calibrated_data_check-postcali.ipynb` - Post-calibration data checks
- `museek_flags.ipynb` - MuSEEK flagging workflow

## Creating New Notebooks

When creating new notebooks for use with `museek_run_notebook`, ensure they:

1. Are compatible with papermill (use parameters in the first code cell with a `parameters` tag)
2. Can run non-interactively (no user input required)
3. Have appropriate error handling
4. Save outputs to files rather than just displaying them

Example parameter cell:
```python
# Parameters (tagged with 'parameters' in notebook metadata)
block_name = "1708972386"  # Default value
box = "6"  # Default value
data_path = "/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline"
```
