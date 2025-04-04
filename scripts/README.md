# Q-Filters Evaluation Scripts

This directory contains scripts for evaluating Q-Filters and K-norm KV cache compression methods on the RULER benchmark.

## Available Scripts

- `run_ruler_benchmark.py`: Core script to run RULER benchmark evaluations with Q-Filters or K-norm
- `evaluate_ruler.sh`: Wrapper script for easy running of evaluations with different configurations
- `compare_results.py`: Script to compare results from different compression methods

## Usage

### Running an Evaluation

To run an evaluation with default settings:

```bash
./scripts/evaluate_ruler.sh --model meta-llama/Llama-2-7b-hf --k_norm
```

This will run the RULER benchmark on Llama-2-7b with k norm compression.

### Available Options

The wrapper script `evaluate_ruler.sh` accepts the following options:

```
--model MODEL_NAME        Model name or path (default: meta-llama/Llama-2-7b-hf)
--ratio RATIO             Compression ratio (default: 8)
--max_seq_length LENGTH   Maximum sequence length (default: 8192) 
--output_dir DIR          Output directory (default: results)
--batch_size SIZE         Batch size (default: 1)
--device DEVICE           Device to run on (default: cuda:0)
--k_norm                  Use K-norm compression
--baseline                Run baseline without compression
--help                    Show this help message
```