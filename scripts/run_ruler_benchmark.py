#!/usr/bin/env python3
# Run RULER benchmark for Q-Filters and K-norm comparison

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_eval import evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hf_cache import QFiltersCache, KNormCache

def parse_args():
    parser = argparse.ArgumentParser(description="Run RULER benchmark with various KV cache compression methods")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--compression_ratio", type=int, default=8, help="Compression ratio for KV cache (default: 8)")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length for evaluation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--use_q_filters", action="store_true", help="Use Q-Filters compression")
    parser.add_argument("--use_k_norm", action="store_true", help="Use K-norm compression")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")
    return parser.parse_args()

def setup_model(args, compression_method):
    """
    Set up the model with the specified KV cache compression method.
    """
    # Create caching configurations
    window_length = args.max_seq_length // args.compression_ratio
    max_length = args.max_seq_length
    
    # Initialize model and tokenizer
    print(f"Loading model {args.model_name}...")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    
    if compression_method == "q_filters":
        print(f"Using Q-Filters compression with ratio {args.compression_ratio}x")
        model.past_key_values = QFiltersCache(
            max_length=max_length,
            window_length=window_length,
            model_name=args.model_name
        )
    elif compression_method == "k_norm":
        print(f"Using K-norm compression with ratio {args.compression_ratio}x")
        model.past_key_values = KNormCache(
            max_length=max_length,
            window_length=window_length
        )
    else:
        print("Using no compression (standard KV cache)")
    
    return model, tokenizer

def run_evaluation(args, model, tokenizer, method_name):
    print(f"Running evaluation for method: {method_name}")
    
    # Use the ruler task with the specified max_seq_length
    task_name = f"ruler:{args.max_seq_length}"
    
    # Run the evaluation
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=f"pretrained={args.model_name},tokenizer={args.model_name}",
        tasks=[task_name],
        num_fewshot=0,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=True
    )
    
    # Save the results
    output_path = os.path.join(args.output_dir, f"{method_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return results

def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluations for specified compression methods
    results = {}
    
    try:
        if args.use_q_filters:
            model, tokenizer = setup_model(args, "q_filters")
            try:
                results["q_filters"] = run_evaluation(args, model, tokenizer, "q_filters")
            except Exception as e:
                print(f"Error during Q-Filters evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        if args.use_k_norm:
            model, tokenizer = setup_model(args, "k_norm")
            try:
                results["k_norm"] = run_evaluation(args, model, tokenizer, "k_norm")
            except Exception as e:
                print(f"Error during K-norm evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        # Run evaluation with no compression for baseline if neither method is specified
        if not (args.use_q_filters or args.use_k_norm):
            model, tokenizer = setup_model(args, "none")
            try:
                results["none"] = run_evaluation(args, model, tokenizer, "none")
            except Exception as e:
                print(f"Error during baseline evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        print("All evaluations completed.")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")

if __name__ == "__main__":
    main() 