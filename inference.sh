#!/bin/bash

# Activate conda environment
source /root/miniconda3/bin/activate mdm4ip

# Default values
DEFAULT_RUN_NAME="DisperSE_VE_06040743"
DEFAULT_NFE=3
DEFAULT_METRICS_ONLY=false
DEFAULT_SOLVER="SDE"
DEFAULT_LOCAL_RANK=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --nfe)
            NFE="$2"
            shift 2
            ;;
        --solver)
            SOLVER="$2"
            shift 2
            ;;
        --local_rank)
            LOCAL_RANK="$2"
            shift 2
            ;;
        --metrics_only)
            METRICS_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
NFE=${NFE:-$DEFAULT_NFE}
METRICS_ONLY=${METRICS_ONLY:-$DEFAULT_METRICS_ONLY}
SOLVER=${SOLVER:-$DEFAULT_SOLVER}
LOCAL_RANK=${LOCAL_RANK:-$DEFAULT_LOCAL_RANK}

# Create log directory
LOG_DIR="/root/autodl-tmp/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# Generate timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/inference_NFE${NFE}_${TIMESTAMP}.log"

echo "Running with parameters:" | tee -a "${LOG_FILE}"
echo "RUN_NAME: ${RUN_NAME}" | tee -a "${LOG_FILE}"
echo "SAMPLING STEPS: ${NFE}" | tee -a "${LOG_FILE}"
echo "SOLVER: ${SOLVER}" | tee -a "${LOG_FILE}"
echo "LOCAL_RANK: ${LOCAL_RANK}" | tee -a "${LOG_FILE}"
echo "METRICS_ONLY: ${METRICS_ONLY}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Run inference unless metrics_only is true
if ! ${METRICS_ONLY}; then
    {
    echo "===== Starting inference ====="
    python -m cli.inference \
        --model_dir "/root/autodl-fs/papers/AAAI2026_RSB/runs/${RUN_NAME}/" \
        --audio_path "/root/autodl-tmp/datasets/Voicebank+Demand/test/noisy/" \
        --output_dir "/root/autodl-tmp/results/${RUN_NAME}/${SOLVER}_num_step=${NFE}" \
        --num_step="${NFE}" \
        --solver="${SOLVER}" \
        --local_rank="${LOCAL_RANK}"
    echo "===== Inference completed ====="
    } 2>&1 | tee -a "${LOG_FILE}"
else
    echo "Skipping inference as --metrics_only flag is set" | tee -a "${LOG_FILE}"
fi

# Always run metrics calculation
{
echo "===== Starting metric calculation ====="
python -m cli.calc_metric \
    --clean_dir "/root/autodl-tmp/datasets/Voicebank+Demand/test/clean/" \
    --noisy_dir "/root/autodl-tmp/datasets/Voicebank+Demand/test/noisy/" \
    --enhanced_dir "/root/autodl-tmp/results/${RUN_NAME}/${SOLVER}_num_step=${NFE}"
echo "===== Metric calculation completed ====="
} 2>&1 | tee -a "${LOG_FILE}"

echo "All operations completed. Log saved to ${LOG_FILE}" | tee -a "${LOG_FILE}"