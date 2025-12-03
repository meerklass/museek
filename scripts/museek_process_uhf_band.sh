#!/bin/bash

################################################################################
# MuSEEK UHF Band Processing Script
# 
# DESCRIPTION:
#   Generates and submits a Slurm job to process UHF band data using the MuSEEK 
#   pipeline. This script creates a temporary sbatch script and submits it to Slurm.
#   This script is designed for processing MEERKLASS observations.
#
# USAGE:
#   museek_process_uhf_band.sh --block-name <block_name> --box <box_number> 
#                             [--base-context-folder <path>] [--data-folder <path>]
#                             [--slurm-options <options>] [--dry-run]
#
# OPTIONS:
#   --block-name <block_name>
#       (required) Block name or observation ID (e.g., 1675632179)
#
#   --box <box_number>
#       (required) Box number of this block name (e.g., 6)
# 
#   --base-context-folder <path>
#       (optional) Path to the base context/output folder
#       The final context folder will be <base-context-folder>/BOX<box>/<block-name>
#       Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline
#
#   --data-folder <path>
#       (optional) Path to raw data folder
#       Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw
#
#   --slurm-options <options>
#       (optional) Additional SLURM options to pass to sbatch
#       Each --slurm-options takes ONE flag (e.g., --mail-user=user@domain.com)
#       Multiple --slurm-options can be specified for multiple flags
#       Examples: 
#         Single: --slurm-options --time=72:00:00
#         Multiple: --slurm-options --mail-user=user@domain.com --slurm-options --mail-type=ALL --slurm-options --time=72:00:00
#
#   --dry-run
#       (optional) Show the generated sbatch script without submitting
#
#   --help
#       Display this help message
#
# EXAMPLES:
#   # Basic usage with box number
#   museek_process_uhf_band.sh --block-name 1675632179 --box 6
#
#   # Custom base context folder
#   museek_process_uhf_band.sh --block-name 1675632179 --box 6 --base-context-folder /custom/path/pipeline
#
#   # Custom base context and data folders
#   museek_process_uhf_band.sh --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline --data-folder /custom/data
#
#   # With SLURM email notifications
#   museek_process_uhf_band.sh --block-name 1675632179 --box 6 \
#       --slurm-options --mail-user=user@uni.edu --mail-type=ALL
#
#   # Dry-run to preview the script
#   museek_process_uhf_band.sh --block-name 1675632179 --box 6 --dry-run
#
#   # Display help
#   museek_process_uhf_band.sh --help
# 
# DEFAULT SLURM PARAMETERS:
#   Job name:       MuSEEK-<block_name>
#   Tasks:          1
#   CPUs per task:  32
#   Memory:         248GB
#   Max time:       48 hours
#   Output:         museek-<block_name>-stdout.log
#   Error:          museek-<block_name>-stderr.log
#
# REQUIREMENTS:
#   - Access to Ilifu
#   - meerklass-1 group permission for raw data access
#   - sbatch command available (Slurm)
################################################################################


# Default values
BLOCK_NAME=""
BOX=""
BASE_CONTEXT_FOLDER="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline"
DATA_FOLDER="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw"
DRY_RUN=false
declare -a SLURM_OPTIONS

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --block-name)
            BLOCK_NAME="$2"
            shift 2
            ;;
        --box)
            BOX="$2"
            shift 2
            ;;
        --base-context-folder)
            BASE_CONTEXT_FOLDER="$2"
            shift 2
            ;;
        --data-folder)
            DATA_FOLDER="$2"
            shift 2
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
MuSEEK UHF Band Processing Script

This script generates and submits a Slurm job to process UHF band data using 
the MuSEEK pipeline.

USAGE:
  museek_process_uhf_band.sh --block-name <block_name> --box <box_number> 
                            [--base-context-folder <path>] [--data-folder <path>]
                            [--slurm-options <options>] [--dry-run]

OPTIONS:
  --block-name <block_name>
      (required) Block name or observation ID (e.g., 1675632179)

  --box <box_number>
      (required) Box number of this block name (e.g., 6)

  --base-context-folder <path>
      (optional) Path to the base context/output folder
      The final context folder will be <base-context-folder>/BOX<box>/<block-name>
      Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline

  --data-folder <path>
      (optional) Path to raw data folder
      Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw

  --slurm-options <options>
      (optional) Additional SLURM options to pass to sbatch
      Each --slurm-options takes ONE flag (e.g., --mail-user=user@domain.com)
      Multiple --slurm-options can be specified for multiple flags
      Examples: 
        Single: --slurm-options --time=72:00:00
        Multiple: --slurm-options --mail-user=user@domain.com --slurm-options --mail-type=ALL

  --dry-run
      (optional) Show the generated sbatch script without submitting

  --help
      Display this help message

EXAMPLES:
  museek_process_uhf_band.sh --block-name 1675632179 --box 6
  museek_process_uhf_band.sh --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline
  museek_process_uhf_band.sh --block-name 1675632179 --box 6 --dry-run
  museek_process_uhf_band.sh --block-name 1675632179 --box 6 --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL --slurm-options --time=72:00:00
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

# Construct the context folder path
CONTEXT_FOLDER="${BASE_CONTEXT_FOLDER}/BOX${BOX}/${BLOCK_NAME}"

# Generate temporary sbatch script
TEMP_SBATCH=$(mktemp)
trap "rm -f $TEMP_SBATCH" EXIT

cat > "$TEMP_SBATCH" << 'SBATCH_SCRIPT'
#!/bin/bash

#SBATCH --job-name='MuSEEK-BLOCK_NAME_PLACEHOLDER'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=248GB
#SBATCH --time=48:00:00
SBATCH_SCRIPT

# Substitute the block name in job name
sed -i "s/BLOCK_NAME_PLACEHOLDER/$BLOCK_NAME/" "$TEMP_SBATCH"

# Add dynamic output/error file names
echo "#SBATCH --output=museek-${BLOCK_NAME}-stdout.log" >> "$TEMP_SBATCH"
echo "#SBATCH --error=museek-${BLOCK_NAME}-stderr.log" >> "$TEMP_SBATCH"

# Add any additional SLURM options
for option in "${SLURM_OPTIONS[@]}"; do
    echo "#SBATCH $option" >> "$TEMP_SBATCH"
done

cat >> "$TEMP_SBATCH" << 'SBATCH_SCRIPT'

# Use shared meerklass environment
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
echo "Python executable is: $(which python)"

# Log job information
echo "=========================================="
echo "Executing MuSEEK UHF Band Processing"
echo "=========================================="
echo "Block name: $BLOCK_NAME"
echo "Box: $BOX"
echo "Data folder: $DATA_FOLDER"
echo "Context folder: $CONTEXT_FOLDER"
echo "=========================================="

# Execute the pipeline
museek --InPlugin-block-name=$BLOCK_NAME --InPlugin-context-folder=$CONTEXT_FOLDER \
    --InPlugin-data-folder=$DATA_FOLDER museek.config.process_uhf_band

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "MuSEEK pipeline completed successfully!"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "Error: MuSEEK pipeline failed!"
    echo "=========================================="
    exit 1
fi
SBATCH_SCRIPT

# Substitute variables in the script
sed -i "s|\$BLOCK_NAME|$BLOCK_NAME|g" "$TEMP_SBATCH"
sed -i "s|\$BOX|$BOX|g" "$TEMP_SBATCH"
sed -i "s|\$DATA_FOLDER|$DATA_FOLDER|g" "$TEMP_SBATCH"
sed -i "s|\$CONTEXT_FOLDER|$CONTEXT_FOLDER|g" "$TEMP_SBATCH"

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
