#!/bin/bash

################################################################################
# MuSEEK Notebook Execution Script
# 
# DESCRIPTION:
#   Generates and submits a Slurm job to execute a MuSEEK Jupyter notebook 
#   using papermill. This script creates a temporary sbatch script and submits 
#   it to Slurm. Kernel validation is performed before submission.
#
# USAGE:
#   ./run_notebook.sh --notebook <notebook_name> --block-name <block_name> --box <box_number>
#                     [--output-path <path>] [--kernel <kernel_name>]
#                     [-p <param_name> <param_value>] ... [-p <param_name> <param_value>]
#                     [--slurm-options <options>] ... [--slurm-options <options>]
#                     [--dry-run]
#
# OPTIONS:
#   --notebook <notebook_name>
#       (required) Name of the notebook to run (e.g., calibrated_data_check-postcali)
#       Available: calibrated_data_check-postcali, calibrated_data_check_observers, etc.
#
#   --block-name <block_name>
#       (required) Block name or observation ID (e.g., 1708972386)
#
#   --box <box_number>
#       (required) Box number of this block name (e.g., 6)
#
#   --output-path <path>
#       (optional) Base directory for notebook output
#       The final output folder will be <output_path>/BOX<box>/<block_name>/
#       Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline
#
#   --kernel <kernel_name>
#       (optional) Jupyter kernel to use for execution
#       Default: meerklass
#
#   -p <param_name> <param_value> | --parameters <param_name> <param_value>
#       (optional, repeatable) Parameters to pass to the notebook via papermill
#       These override notebook defaults
#       Examples: -p block_name 1708972386 -p data_path /custom/path/
#
#   --slurm-options <options>
#       (optional) Additional SLURM options to pass to sbatch
#       Each --slurm-options takes ONE flag (e.g., --mail-user=user@domain.com)
#       Multiple --slurm-options can be specified for multiple flags
#       Examples: 
#         Single: --slurm-options --time=02:00:00
#         Multiple: --slurm-options --mail-user=user@domain.com --slurm-options --mail-type=ALL
#
#   --dry-run
#       (optional) Show the generated sbatch script without submitting
#
#   --help
#       Display this help message
#
# EXAMPLES:
#   # Basic usage
#   ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6
#
#   # With custom parameters
#   ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 \
#       -p data_path /custom/path/ -p data_name custom.pickle
#
#   # With SLURM options
#   ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 \
#       --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL
#
#   # Dry-run to preview the script
#   ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --dry-run
#
#   # Display help
#   ./run_notebook.sh --help
#
# DEFAULT SLURM PARAMETERS:
#   Job name:       MuSEEK-Notebook
#   Tasks:          1
#   CPUs per task:  32
#   Memory:         248GB
#   Max time:       1 hour
#   Output:         notebook-<block_name>-stdout.log
#   Error:          notebook-<block_name>-stderr.log
#
# REQUIREMENTS:
#   - Access to Ilifu
#   - Jupyter kernel installed or auto-installable (meerklass)
#   - papermill installed
#   - sbatch command available (Slurm)
#
################################################################################


# Default values
NOTEBOOK=""
BLOCK_NAME=""
BOX=""
OUTPUT_PATH="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline"
KERNEL="meerklass"
DRY_RUN=false
declare -a PAPERMILL_PARAMS
declare -a SLURM_OPTIONS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOKS_DIR="$PROJECT_ROOT/notebooks"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --notebook)
            NOTEBOOK="$2"
            shift 2
            ;;
        --block-name)
            BLOCK_NAME="$2"
            shift 2
            ;;
        --box)
            BOX="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --kernel)
            KERNEL="$2"
            shift 2
            ;;
        -p|--parameters)
            PAPERMILL_PARAMS+=("-p" "$2" "$3")
            shift 3
            ;;
        --slurm-options)
            SLURM_OPTIONS+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            cat << 'EOF'
MuSEEK Notebook Execution Script

This script generates and submits a Slurm job to execute a MuSEEK Jupyter 
notebook using papermill.

USAGE:
  ./run_notebook.sh --notebook <notebook_name> --block-name <block_name> --box <box_number>
                    [--output-path <path>] [--kernel <kernel_name>]
                    [-p <param_name> <param_value>] ... [-p <param_name> <param_value>]
                    [--slurm-options <options>] ... [--slurm-options <options>] 
                    [--dry-run]

OPTIONS:
  --notebook <notebook_name>
      (required) Name of the notebook to run (e.g., calibrated_data_check-postcali)

  --block-name <block_name>
      (required) Block name or observation ID (e.g., 1708972386)

  --box <box_number>
      (required) Box number of this block name (e.g., 6)

  --output-path <path>
      (optional) Base directory for notebook output
      The final output folder will be <output_path>/BOX<box>/<block_name>/
      Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline

  --kernel <kernel_name>
      (optional) Jupyter kernel to use for execution
      Default: meerklass

  -p <param_name> <param_value> | --parameters <param_name> <param_value>
      (optional, repeatable) Parameters to pass to the notebook via papermill
      These override notebook defaults
      Examples: -p data_path /custom/path/ -p data_name custom.pickle

  --slurm-options <options>
      (optional) Additional SLURM options to pass to sbatch
      Each --slurm-options takes ONE flag (e.g., --mail-user=user@domain.com)
      Multiple --slurm-options can be specified for multiple flags
      Examples: 
        Single: --slurm-options --time=02:00:00
        Multiple: --slurm-options --mail-user=user@domain.com --slurm-options --mail-type=ALL

  --dry-run
      (optional) Show the generated sbatch script without submitting

  --help
      Display this help message

EXAMPLES:
  ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6
  ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 -p data_path /custom/path/
  ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --dry-run
  ./run_notebook.sh --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$NOTEBOOK" ]; then
    echo "Error: --notebook is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$BLOCK_NAME" ]; then
    echo "Error: --block-name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$BOX" ]; then
    echo "Error: --box is required"
    echo "Use --help for usage information"
    exit 1
fi

# Use shared meerklass environment
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
echo "Python executable is: $(which python)"

# Check if provided kernel is present and install if needed
KERNEL_LIST=$(jupyter kernelspec list 2>/dev/null | grep -E "^\s*${KERNEL}" || true)

if [ -z "$KERNEL_LIST" ]; then
    echo "Warning: Kernel '${KERNEL}' not found in jupyter kernelspec list"
    
    if [ "$KERNEL" = "meerklass" ]; then
        echo "Installing meerklass kernel..."
        python -m ipykernel install --name "meerklass" --user
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install meerklass kernel"
            exit 1
        fi
        echo "Successfully installed meerklass kernel"
    else
        echo "Error: Kernel '${KERNEL}' is not available and will not be auto-installed"
        echo "Please install the kernel or specify a different kernel with --kernel option"
        echo "Available kernels:"
        jupyter kernelspec list
        exit 1
    fi
else
    echo "Kernel '${KERNEL}' is available"
fi

# Verify notebook exists
NOTEBOOK_PATH="$NOTEBOOKS_DIR/${NOTEBOOK}.ipynb"
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: Notebook not found: $NOTEBOOK_PATH"
    echo "Available notebooks in $NOTEBOOKS_DIR:"
    ls -1 "$NOTEBOOKS_DIR"/*.ipynb 2>/dev/null | xargs -I {} basename {} .ipynb || echo "No notebooks found"
    exit 1
fi

# Generate temporary sbatch script
TEMP_SBATCH=$(mktemp)
trap "rm -f $TEMP_SBATCH" EXIT

cat > "$TEMP_SBATCH" << 'SBATCH_SCRIPT'
#!/bin/bash

#SBATCH --job-name='MuSEEK-Notebook'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=248GB
#SBATCH --time=01:00:00
SBATCH_SCRIPT

# Add dynamic output/error file names
echo "#SBATCH --output=notebook-${BLOCK_NAME}-stdout.log" >> "$TEMP_SBATCH"
echo "#SBATCH --error=notebook-${BLOCK_NAME}-stderr.log" >> "$TEMP_SBATCH"

# Add any additional SLURM options
for option in "${SLURM_OPTIONS[@]}"; do
    echo "#SBATCH $option" >> "$TEMP_SBATCH"
done

cat >> "$TEMP_SBATCH" << 'SBATCH_SCRIPT'

# Use shared meerklass environment
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
echo "Python executable is: $(which python)"
echo "Papermill version: $(papermill --version)"

# Log job information
echo "=========================================="
echo "Executing MuSEEK Notebook"
echo "=========================================="
echo "Notebook:      NOTEBOOK_PATH_PLACEHOLDER"
echo "Block name:    BLOCK_NAME_PLACEHOLDER"
echo "Box:           BOX_PLACEHOLDER"
echo "Kernel:        KERNEL_PLACEHOLDER"
echo "Output:        OUTPUT_NOTEBOOK_PLACEHOLDER"
echo "=========================================="

# Execute notebook using papermill with collected parameters
papermill -k "KERNEL_PLACEHOLDER" \
    PAPERMILL_PARAMS_PLACEHOLDER \
    "NOTEBOOK_PATH_PLACEHOLDER" "OUTPUT_NOTEBOOK_PLACEHOLDER"

# Check if papermill executed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Notebook executed successfully!"
    echo "Output saved to: OUTPUT_NOTEBOOK_PLACEHOLDER"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "Error: Notebook execution failed!"
    echo "Check the output files for details"
    echo "=========================================="
    exit 1
fi
SBATCH_SCRIPT

# Construct output directory with BOX structure
OUTPUT_DIR="${OUTPUT_PATH}/BOX${BOX}/${BLOCK_NAME}"
OUTPUT_NOTEBOOK="${OUTPUT_DIR}/${NOTEBOOK}_output.ipynb"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Could not create output directory: $OUTPUT_DIR"
    exit 1
fi

# Substitute variables in the script
sed -i "s|NOTEBOOK_PATH_PLACEHOLDER|$NOTEBOOK_PATH|g" "$TEMP_SBATCH"
sed -i "s|BLOCK_NAME_PLACEHOLDER|$BLOCK_NAME|g" "$TEMP_SBATCH"
sed -i "s|BOX_PLACEHOLDER|$BOX|g" "$TEMP_SBATCH"
sed -i "s|KERNEL_PLACEHOLDER|$KERNEL|g" "$TEMP_SBATCH"
sed -i "s|OUTPUT_NOTEBOOK_PLACEHOLDER|$OUTPUT_NOTEBOOK|g" "$TEMP_SBATCH"

# Substitute papermill parameters
if [ ${#PAPERMILL_PARAMS[@]} -gt 0 ]; then
    PARAMS_STR=""
    for ((i=0; i<${#PAPERMILL_PARAMS[@]}; i++)); do
        PARAMS_STR+="\"${PAPERMILL_PARAMS[$i]}\" "
    done
    sed -i "s|PAPERMILL_PARAMS_PLACEHOLDER|$PARAMS_STR|g" "$TEMP_SBATCH"
else
    sed -i "s|PAPERMILL_PARAMS_PLACEHOLDER||g" "$TEMP_SBATCH"
fi

# Display or submit the script
if [ "$DRY_RUN" = true ]; then
    echo "=========================================="
    echo "DRY-RUN: Generated sbatch script"
    echo "=========================================="
    cat "$TEMP_SBATCH"
    echo "=========================================="
    echo "To submit this job, run without --dry-run"
    echo "=========================================="
else
    echo "=========================================="
    echo "Submitting job to Slurm"
    echo "=========================================="
    sbatch "$TEMP_SBATCH"
    if [ $? -eq 0 ]; then
        echo "Job submitted successfully!"
    else
        echo "Error: Failed to submit job"
        exit 1
    fi
fi
