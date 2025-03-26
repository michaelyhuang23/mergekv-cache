#!/bin/bash
# Wrapper script to evaluate models on RULER benchmark with different compression methods

# Default parameters
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
COMPRESSION_RATIO=8
MAX_SEQ_LENGTH=8192
OUTPUT_DIR="results"
BATCH_SIZE=1
DEVICE="cuda:0"

# Add the current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --ratio)
      COMPRESSION_RATIO="$2"
      shift 2
      ;;
    --max_seq_length)
      MAX_SEQ_LENGTH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --q_filters)
      USE_Q_FILTERS=true
      shift
      ;;
    --k_norm)
      USE_K_NORM=true
      shift
      ;;
    --baseline)
      USE_BASELINE=true
      shift
      ;;
    --all)
      USE_Q_FILTERS=true
      USE_K_NORM=true
      USE_BASELINE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model MODEL_NAME        Model name or path (default: meta-llama/Llama-2-7b-hf)"
      echo "  --ratio RATIO             Compression ratio (default: 8)"
      echo "  --max_seq_length LENGTH   Maximum sequence length (default: 4096)"
      echo "  --output_dir DIR          Output directory (default: results)"
      echo "  --batch_size SIZE         Batch size (default: 1)"
      echo "  --device DEVICE           Device to run on (default: cuda:0)"
      echo "  --q_filters               Use Q-Filters compression"
      echo "  --k_norm                  Use K-norm compression"
      echo "  --baseline                Run baseline without compression"
      echo "  --all                     Run all compression methods"
      echo "  --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Make results directory
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/evaluation_${TIMESTAMP}.log"

echo "=== RULER Benchmark Evaluation ===" | tee -a "$LOG_FILE"
echo "Model: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Compression ratio: ${COMPRESSION_RATIO}x" | tee -a "$LOG_FILE"
echo "Max sequence length: $MAX_SEQ_LENGTH" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run evaluations based on specified methods
CMD_ARGS="--model_name $MODEL_NAME --compression_ratio $COMPRESSION_RATIO --max_seq_length $MAX_SEQ_LENGTH --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --device $DEVICE"

if [ "$USE_Q_FILTERS" = true ]; then
  echo "Running evaluation with Q-Filters compression..." | tee -a "$LOG_FILE"
  python scripts/run_ruler_benchmark.py $CMD_ARGS --use_q_filters 2>&1 | tee -a "$LOG_FILE"
fi

if [ "$USE_K_NORM" = true ]; then
  echo "Running evaluation with K-norm compression..." | tee -a "$LOG_FILE"
  python scripts/run_ruler_benchmark.py $CMD_ARGS --use_k_norm 2>&1 | tee -a "$LOG_FILE"
fi

if [ "$USE_BASELINE" = true ]; then
  echo "Running baseline evaluation without compression..." | tee -a "$LOG_FILE"
  python scripts/run_ruler_benchmark.py $CMD_ARGS 2>&1 | tee -a "$LOG_FILE"
fi

# If no method is specified, run all
if [ -z "$USE_Q_FILTERS" ] && [ -z "$USE_K_NORM" ] && [ -z "$USE_BASELINE" ]; then
  echo "No specific compression method specified, running both Q-Filters and K-norm..." | tee -a "$LOG_FILE"
  python scripts/run_ruler_benchmark.py $CMD_ARGS --use_q_filters 2>&1 | tee -a "$LOG_FILE"
  python scripts/run_ruler_benchmark.py $CMD_ARGS --use_k_norm 2>&1 | tee -a "$LOG_FILE"
fi

echo "Evaluation completed. Results saved to $OUTPUT_DIR" | tee -a "$LOG_FILE" 