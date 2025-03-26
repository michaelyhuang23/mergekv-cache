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
./scripts/evaluate_ruler.sh --model meta-llama/Llama-2-7b-hf --q_filters
```

This will run the RULER benchmark on Llama-2-7b with Q-Filters compression.

### Available Options

The wrapper script `evaluate_ruler.sh` accepts the following options:

```
--model MODEL_NAME        Model name or path (default: meta-llama/Llama-2-7b-hf)
--ratio RATIO             Compression ratio (default: 8)
--max_seq_length LENGTH   Maximum sequence length (default: 8192) 
--output_dir DIR          Output directory (default: results)
--batch_size SIZE         Batch size (default: 1)
--device DEVICE           Device to run on (default: cuda:0)
--q_filters               Use Q-Filters compression
--k_norm                  Use K-norm compression
--baseline                Run baseline without compression
--all                     Run all compression methods
--help                    Show this help message
```

### Comparing Results

Once you have multiple evaluation results, you can compare them:

```bash
./scripts/compare_results.py --result_dirs results/q_filters_8x_* results/k_norm_8x_* --output_dir comparisons --plot
```

This will create a comparison table and plots for the specified result directories.

## Example Workflow

1. Generate Q-Filters for your model (if not available):

```bash
python make_filters.py --model_name meta-llama/Llama-2-7b-hf --model_cls LlamaForCausalLM --dataset_name PatrickHaller/fineweb-1B --save_mode disk --save_dir ./filters
```

2. Run evaluations with different compression ratios:

```bash
./scripts/evaluate_ruler.sh --model meta-llama/Llama-2-7b-hf --q_filters --ratio 8
./scripts/evaluate_ruler.sh --model meta-llama/Llama-2-7b-hf --k_norm --ratio 8
./scripts/evaluate_ruler.sh --model meta-llama/Llama-2-7b-hf --baseline
```

3. Compare the results:

```bash
./scripts/compare_results.py --result_dirs results/* --output_dir comparisons --plot
```

4. View the comparison results in the `comparisons` directory. 