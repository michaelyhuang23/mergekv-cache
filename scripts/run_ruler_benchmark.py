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
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hf_cache import KNormCache

def parse_args():
    parser = argparse.ArgumentParser(description="Run RULER benchmark with various KV cache compression methods")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HuggingFace model name or path")
    parser.add_argument("--compression_ratio", type=int, default=8, help="Compression ratio for KV cache (default: 8)")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length for evaluation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--use_k_norm", action="store_true", help="Use K-norm compression")
    parser.add_argument("--use_baseline", action="store_true", help="Use baseline (no compression)")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    
    # Create a custom HFLM wrapper class that applies KV cache during forward pass
    class CustomCacheHFLM(HFLM):
        def __init__(self, cache_type, pretrained=None, tokenizer=None, batch_size=1):
            super().__init__(pretrained=pretrained, tokenizer=tokenizer, batch_size=batch_size)
            self.cache_type = cache_type
            
            # Initialize appropriate cache based on method
            if cache_type == "k_norm":
                print(f"Using K-norm compression with ratio {args.compression_ratio}x")
                self.kv_cache = KNormCache(
                    max_length=max_length,
                    window_length=window_length
                )
            elif cache_type == "baseline":
                print("Using no compression (standard KV cache)")
                self.kv_cache = None
            else:
                raise ValueError(f"Invalid cache type: {cache_type}")
        
        def generate(self, *args, **kwargs):
            """Override the _model_call method to use our custom KV cache"""
            return self.model.generate(*args, **kwargs, past_key_values=self.kv_cache)
        
        def forward(self, *args, **kwargs):
            """Override the forward method to use our custom KV cache"""
            return self.model.forward(*args, **kwargs, past_key_values=self.kv_cache)
    
    # Create the appropriate model wrapper
    if compression_method in ["k_norm"]:
        model_wrapper = CustomCacheHFLM(
            cache_type=compression_method,
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size
        )
    else:
        # For baseline, use the standard HFLM wrapper
        model_wrapper = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size
        )
    
    return model_wrapper, model, tokenizer

def run_evaluation(args, model_wrapper, model, tokenizer, method_name):
    print(f"Running evaluation for method: {method_name}")
    
    # Specify which RULER task to run (based on sequence length)
    task_name = "ruler"
    
    # Run the evaluation
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        model_args=f"max_seq_length={32768},tokenizer={args.model_name},pretrained={args.model_name}",
        tasks=[task_name],
        num_fewshot=0,
        batch_size=args.batch_size,
        metadata={"max_seq_lengths":[4096,8192,16384]},
        device=args.device,
    )

    # Save the results
    output_path = os.path.join(args.output_dir, f"{method_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Cleanup to prevent memory leaks
    del model_wrapper
    
    return results

def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluations for specified compression methods
    results = {}
    
    try:
        if args.use_k_norm:
            model_wrapper, model, tokenizer = setup_model(args, "k_norm")
            try:
                results["k_norm"] = run_evaluation(args, model_wrapper, model, tokenizer, "k_norm")
            except Exception as e:
                print(f"Error during K-norm evaluation: {e}")
            finally:
                # Free up GPU memory
                del model_wrapper
                del model
                torch.cuda.empty_cache()
        
        # Run evaluation with no compression for baseline if neither method is specified
        if args.use_baseline:
            model_wrapper, model, tokenizer = setup_model(args, "baseline")
            try:
                results["baseline"] = run_evaluation(args, model_wrapper, model, tokenizer, "baseline")
            except Exception as e:
                print(f"Error during baseline evaluation: {e}")
            finally:
                # Free up GPU memory
                del model_wrapper
                del model
                torch.cuda.empty_cache()
        
        print("All evaluations completed.")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 