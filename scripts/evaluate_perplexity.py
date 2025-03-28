#!/usr/bin/env python3
# Evaluate language modeling perplexity with various KV cache compression methods

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.hf_cache import QFiltersCache, KNormCache

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate language modeling perplexity with KV cache compression")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--compression_ratio", type=int, default=8, help="Compression ratio for KV cache (default: 8)")
    parser.add_argument("--kv_cache_size", type=int, default=512, help="Maximum KV cache size (default: 512)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for evaluation")
    parser.add_argument("--output_dir", type=str, default="perplexity_results", help="Directory to save results")
    parser.add_argument("--use_q_filters", action="store_true", help="Use Q-Filters compression")
    parser.add_argument("--use_k_norm", action="store_true", help="Use K-norm compression")
    parser.add_argument("--use_streaming_llm", action="store_true", help="Use StreamingLLM approach (keeping first tokens)")
    parser.add_argument("--num_sequences", type=int, default=20, help="Number of sequences to evaluate (default: 20)")
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-small", 
                        help="HuggingFace dataset name for evaluation (default: NeelNanda/pile-small)")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")
    parser.add_argument("--plot", action="store_true", help="Generate plots of perplexity results")
    return parser.parse_args()

def setup_model(args, compression_method):
    """
    Set up the model with the specified KV cache compression method.
    """
    # Create caching configurations based on compression method
    window_length = args.kv_cache_size // args.compression_ratio
    max_length = args.kv_cache_size
    
    if compression_method == "streaming_llm":
        # For StreamingLLM: Keep first tokens (attention sink) and last tokens
        window_length = max_length - window_length
    
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
    elif compression_method == "streaming_llm":
        print(f"Using StreamingLLM approach with {window_length} recent tokens and {max_length-window_length} attention sink tokens")
        # For StreamingLLM, we implement a simple version that keeps the first (max_length-window_length) tokens
        # and most recent window_length tokens
        from transformers.cache_utils import DynamicCache, SinkCache
        model.past_key_values = SinkCache(
            window_length=window_length,
            num_sink_tokens=max_length-window_length
        )
    else:
        print("Using no compression (standard KV cache)")
    
    return model, tokenizer

def calculate_perplexity(logits, labels, ignore_index=-100):
    """
    Calculate perplexity from logits and labels.
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Reshape loss back to original shape
    loss = loss.view(shift_labels.size())
    
    # Calculate perplexity from token-wise loss
    perplexity = torch.exp(loss)
    
    # Create a mask for non-ignored tokens
    mask = (shift_labels != ignore_index).float()
    
    return perplexity, mask

def evaluate_perplexity(args, model, tokenizer, method_name):
    """
    Evaluate perplexity on sequences from the dataset.
    """
    print(f"Evaluating perplexity for method: {method_name}")
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    # Determine text field based on dataset
    text_field = "text" if "text" in dataset.column_names else dataset.column_names[0]
    
    # Setup tracking variables
    perplexity_by_position = {}
    max_position = args.max_seq_length
    
    # Track number of samples processed
    num_processed = 0
    
    # Process sequences
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating perplexity")):
            if num_processed >= args.num_sequences:
                break
                
            # Get text from sample
            text = sample[text_field]
            if not isinstance(text, str) or len(text) < 100:  # Skip very short texts
                continue
                
            # Truncate to max_seq_length
            encoded = tokenizer(text, return_tensors="pt", truncation=True, 
                                max_length=args.max_seq_length)
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            
            # Skip sequences that are too short
            if input_ids.size(1) < 100:
                continue
                
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits
            
            # Calculate perplexity
            perplexity, mask = calculate_perplexity(logits, input_ids)
            
            # Aggregate perplexity by position
            seq_length = perplexity.size(1)
            for pos in range(seq_length):
                if pos not in perplexity_by_position:
                    perplexity_by_position[pos] = []
                if mask[0, pos] > 0:  # Only include non-ignored tokens
                    perplexity_by_position[pos].append(perplexity[0, pos].item())
            
            # Increment processed count
            num_processed += 1
    
    # Calculate average perplexity by position
    avg_perplexity = {}
    for pos, values in perplexity_by_position.items():
        if values:  # Only calculate if we have values
            avg_perplexity[pos] = sum(values) / len(values)
    
    # Prepare results
    positions = sorted(avg_perplexity.keys())
    perplexities = [avg_perplexity[pos] for pos in positions]
    
    results = {
        "method": method_name,
        "compression_ratio": args.compression_ratio,
        "kv_cache_size": args.kv_cache_size,
        "positions": positions,
        "perplexities": perplexities,
        "average_perplexity": sum(perplexities) / len(perplexities) if perplexities else float('nan'),
        "final_perplexity": perplexities[-1] if perplexities else float('nan')
    }
    
    # Save the results
    output_path = os.path.join(args.output_dir, f"{method_name}_perplexity.json")
    with open(output_path, "w") as f:
        # Convert numpy values to python types for JSON serialization
        serializable_results = {
            "method": results["method"],
            "compression_ratio": results["compression_ratio"],
            "kv_cache_size": results["kv_cache_size"],
            "positions": [int(p) for p in results["positions"]],
            "perplexities": [float(p) for p in results["perplexities"]],
            "average_perplexity": float(results["average_perplexity"]),
            "final_perplexity": float(results["final_perplexity"])
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Average perplexity: {results['average_perplexity']:.4f}")
    print(f"Final perplexity: {results['final_perplexity']:.4f}")
    
    return results

def plot_results(results_list, args):
    """
    Plot perplexity results for all methods.
    """
    plt.figure(figsize=(10, 6))
    
    for results in results_list:
        method = results["method"]
        positions = results["positions"]
        perplexities = results["perplexities"]
        
        # Plot perplexity vs position
        plt.plot(positions, perplexities, label=f"{method} (ratio: {args.compression_ratio}x)")
    
    plt.xlabel("Position in Sequence")
    plt.ylabel("Perplexity")
    plt.title(f"Language Modeling Perplexity with KV Cache Size {args.kv_cache_size}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(args.output_dir, f"perplexity_comparison_r{args.compression_ratio}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    
    # Second plot with logarithmic y-axis for better visualization
    plt.figure(figsize=(10, 6))
    
    for results in results_list:
        method = results["method"]
        positions = results["positions"]
        perplexities = results["perplexities"]
        
        # Plot perplexity vs position with log scale
        plt.semilogy(positions, perplexities, label=f"{method} (ratio: {args.compression_ratio}x)")
    
    plt.xlabel("Position in Sequence")
    plt.ylabel("Perplexity (log scale)")
    plt.title(f"Language Modeling Perplexity with KV Cache Size {args.kv_cache_size} (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the log-scale plot
    log_plot_path = os.path.join(args.output_dir, f"perplexity_comparison_log_r{args.compression_ratio}.png")
    plt.savefig(log_plot_path, dpi=300, bbox_inches="tight")
    print(f"Log-scale plot saved to {log_plot_path}")

def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluations for specified compression methods
    results_list = []
    
    try:
        if args.use_q_filters:
            model, tokenizer = setup_model(args, "q_filters")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "q_filters")
                results_list.append(results)
            except Exception as e:
                print(f"Error during Q-Filters evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        if args.use_k_norm:
            model, tokenizer = setup_model(args, "k_norm")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "k_norm")
                results_list.append(results)
            except Exception as e:
                print(f"Error during K-norm evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        if args.use_streaming_llm:
            model, tokenizer = setup_model(args, "streaming_llm")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "streaming_llm")
                results_list.append(results)
            except Exception as e:
                print(f"Error during StreamingLLM evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        # Run evaluation with no compression for baseline if neither method is specified
        if not (args.use_q_filters or args.use_k_norm or args.use_streaming_llm):
            model, tokenizer = setup_model(args, "none")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "none")
                results_list.append(results)
            except Exception as e:
                print(f"Error during baseline evaluation: {e}")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        # Generate plots if requested and we have results to plot
        if args.plot and results_list:
            plot_results(results_list, args)
        
        print("All evaluations completed.")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")

if __name__ == "__main__":
    main() 