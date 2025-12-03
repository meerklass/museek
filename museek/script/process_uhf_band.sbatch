#!/bin/bash

################################################################################
# MuSEEK UHF Band Processing Script
# 
# DESCRIPTION:
#   Submits a Slurm job to process UHF band data using the MuSEEK pipeline.
#   This script is designed for processing MEERKLASS observations.
#
# USAGE:
#   sbatch process_uhf_band.sbatch --block-name <block_name> --box <box_number> 
#                                    [--base-context-folder <path>] [--data-folder <path>]
#
# OPTIONS:
#   --block-name <block_name>
#       (required) Block name or observation ID (e.g., 1675632179)
#
#   --box <block_name>
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
#   --help
#       Display this help message
#
# EXAMPLES:
#   # Basic usage with box number (constructs context folder automatically)
#   sbatch process_uhf_band.sbatch --block-name 1675632179 --box 6
#
#   # Custom base context folder
#   sbatch process_uhf_band.sbatch --block-name 1675632179 --box 6 --base-context-folder /custom/path/pipeline
#
#   # Custom base context and data folders
#   sbatch process_uhf_band.sbatch --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline --data-folder /custom/data
#
#   # Tell SLURM to send email reports on job status
#   sbatch --mail-user=jane.doe@university.edu --mail-type=ALL process_uhf_band.sbatch \
#       --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline
#
#   # Display help
#    sbatch process_uhf_band.sbatch --help
# 
# DEFAULT SLURM PARAMETERS:
#   Job name:       MuSEEK
#   Tasks:          1
#   CPUs per task:  32
#   Memory:         248GB
#   Max time:       48 hours
#   Output:         museek-stdout.log
#   Error:          museek-stderr.log
#
# REQUIREMENTS:
#   - Access to Ilifu
#   - meerklass-1 group permission for raw data access
################################################################################

#SBATCH --job-name='MuSEEK'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=248GB
#SBATCH --output=museek-stdout.log
#SBATCH --error=museek-stderr.log
#SBATCH --time=48:00:00


# Default values
BLOCK_NAME=""
BOX=""
BASE_CONTEXT_FOLDER="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline"
DATA_FOLDER="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw"
CONTEXT_FOLDER=""

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
        --help)
            cat << 'EOF'
MuSEEK UHF Band Processing Script.

Run MuSEEK with process_uhf_band.py pipeline config, passing in neccesary parameters.

USAGE:
  sbatch process_uhf_band.sbatch --block-name <block_name> --box <box_number> 
                                   [--base-context-folder <path>] [--data-folder <path>]

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

  --help
      Display this help message

EXAMPLES:
  sbatch process_uhf_band.sbatch --block-name 1675632179 --box 6
  sbatch process_uhf_band.sbatch --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline
  sbatch process_uhf_band.sbatch --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline --data-folder /custom/data
  sbatch --mail-user=jane.doe@university.edu --mail-type=ALL process_uhf_band.sbatch --block-name 1675632179 --box 6
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

# Use shared meerklass environment
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
echo "Python executable is: $(which python)"

# Log repository information
echo "Submitting Slurm job"
echo "Block name: $BLOCK_NAME"
echo "Data folder: $DATA_FOLDER"
echo "Context folder: $CONTEXT_FOLDER"

museek --InPlugin-block-name=$BLOCK_NAME --InPlugin-context-folder=$CONTEXT_FOLDER \
    --InPlugin-data-folder=$DATA_FOLDER museek.config.process_uhf_band
