#!/bin/bash
# Wrapper script to evaluate language modeling perplexity with different compression methods

# Default parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
WINDOW_SIZE=512
TOPK=32
MAX_SEQ_LENGTH=2048
OUTPUT_DIR="perplexity_results"
NUM_SEQUENCES=5
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
    --window)
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --topk)
      TOPK="$2"
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
    --num_sequences)
      NUM_SEQUENCES="$2"
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
    --dataset)
      DATASET_NAME="$2"
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
    --streaming_llm)
      USE_STREAMING_LLM=true
      shift
      ;;
    --baseline)
      USE_BASELINE=true
      shift
      ;;
    --all)
      USE_Q_FILTERS=true
      USE_K_NORM=true
      USE_STREAMING_LLM=true
      USE_BASELINE=true
      shift
      ;;
    --plot)
      PLOT=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model MODEL_NAME        Model name or path (default: meta-llama/Llama-3.1-8B-Instruct)"
      echo "  --window SIZE             Window size for recent tokens (default: 512)"
      echo "  --topk SIZE               Number of filtered tokens to keep (default: 32)"
      echo "  --max_seq_length LENGTH   Maximum sequence length (default: 2048)"
      echo "  --output_dir DIR          Output directory (default: perplexity_results)"
      echo "  --num_sequences COUNT     Number of sequences to evaluate (default: 5)"
      echo "  --batch_size SIZE         Batch size (default: 1)"
      echo "  --device DEVICE           Device to run on (default: cuda:0)"
      echo "  --dataset NAME            HuggingFace dataset name (default: NeelNanda/pile-10k)"
      echo "  --q_filters               Use Q-Filters compression"
      echo "  --k_norm                  Use K-norm compression"
      echo "  --streaming_llm           Use StreamingLLM approach"
      echo "  --baseline                Run baseline without compression"
      echo "  --all                     Run all compression methods"
      echo "  --plot                    Generate plots of results"
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
LOG_FILE="${OUTPUT_DIR}/perplexity_eval_${TIMESTAMP}.log"

echo "=== Language Modeling Perplexity Evaluation ===" | tee -a "$LOG_FILE"
echo "Model: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Window size: $WINDOW_SIZE" | tee -a "$LOG_FILE"
echo "TopK: $TOPK" | tee -a "$LOG_FILE"
echo "Max sequence length: $MAX_SEQ_LENGTH" | tee -a "$LOG_FILE"
echo "Number of sequences: $NUM_SEQUENCES" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Build common command arguments
CMD_ARGS="--model_name $MODEL_NAME \
  --window_size $WINDOW_SIZE \
  --topk $TOPK \
  --max_seq_length $MAX_SEQ_LENGTH \
  --output_dir $OUTPUT_DIR \
  --num_sequences $NUM_SEQUENCES \
  --batch_size $BATCH_SIZE \
  --device $DEVICE"

# Add dataset name if specified
if [ ! -z "$DATASET_NAME" ]; then
  CMD_ARGS="$CMD_ARGS --dataset_name $DATASET_NAME"
fi

# Add plot flag if specified
if [ "$PLOT" = true ]; then
  CMD_ARGS="$CMD_ARGS --plot"
fi

# Set up command based on which methods should be evaluated
METHODS_FLAGS=""

if [ "$USE_Q_FILTERS" = true ]; then
  METHODS_FLAGS="$METHODS_FLAGS --use_q_filters"
fi

if [ "$USE_K_NORM" = true ]; then
  METHODS_FLAGS="$METHODS_FLAGS --use_k_norm"
fi

if [ "$USE_STREAMING_LLM" = true ]; then
  METHODS_FLAGS="$METHODS_FLAGS --use_streaming_llm"
fi

if [ "$USE_BASELINE" = true ]; then
  METHODS_FLAGS="$METHODS_FLAGS --use_baseline"
fi

# If no method is specified, run baseline only
if [ -z "$USE_Q_FILTERS" ] && [ -z "$USE_K_NORM" ] && [ -z "$USE_STREAMING_LLM" ] && [ -z "$USE_BASELINE" ]; then
  echo "No specific compression method specified, running baseline evaluation..." | tee -a "$LOG_FILE"
else
  echo "Running evaluation with specified methods..." | tee -a "$LOG_FILE"
  CMD_ARGS="$CMD_ARGS $METHODS_FLAGS"
fi

# Run the evaluation
echo "Command: python scripts/evaluate_perplexity.py $CMD_ARGS" | tee -a "$LOG_FILE"
python scripts/evaluate_perplexity.py $CMD_ARGS 2>&1 | tee -a "$LOG_FILE"

echo "Evaluation completed. Results saved to $OUTPUT_DIR" | tee -a "$LOG_FILE" 