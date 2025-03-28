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
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", 
                        help="HuggingFace model name or path")
    
    # Cache configuration
    parser.add_argument("--window_size", type=int, default=512, 
                        help="Window size for recent tokens to always keep (default: 512)")
    parser.add_argument("--topk", type=int, default=64, 
                        help="Number of filtered tokens to keep beyond window (default: 32)")
    
    # Evaluation parameters
    parser.add_argument("--max_seq_length", type=int, default=4096, 
                        help="Maximum sequence length for evaluation")
    parser.add_argument("--num_sequences", type=int, default=5, 
                        help="Number of sequences to evaluate (default: 5)")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to run evaluation on")
    
    # Dataset options
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k", 
                        help="HuggingFace dataset name for evaluation (default: NeelNanda/pile-10k)")
    parser.add_argument("--dataset_split", type=str, default="train", 
                        help="Dataset split to use")
    
    # Compression methods to evaluate
    parser.add_argument("--use_q_filters", action="store_true", 
                        help="Use Q-Filters compression")
    parser.add_argument("--use_k_norm", action="store_true", 
                        help="Use K-norm compression")
    parser.add_argument("--use_streaming_llm", action="store_true", 
                        help="Use StreamingLLM approach (keeping first tokens)")
    parser.add_argument("--use_baseline", action="store_true", 
                        help="Run baseline without compression")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--plot", action="store_true", 
                        help="Generate plots of perplexity results")
    
    return parser.parse_args()

def setup_model(args, compression_method):
    """
    Set up the model with the specified KV cache compression method.
    """
    # Create caching configurations based on compression method
    window_length = args.window_size
    max_length = args.window_size + args.topk
    
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
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    
    # Create the appropriate cache
    if compression_method == "q_filters":
        print(f"Using Q-Filters compression with sliding window size {window_length} and max length {max_length}")
        kv_cache = QFiltersCache(
            max_length=max_length,
            window_length=window_length,
            model_name=args.model_name
        )
    elif compression_method == "k_norm":
        print(f"Using K-norm compression with window size {window_length} and max length {max_length}")
        kv_cache = KNormCache(
            max_length=max_length,
            window_length=window_length
        )
    elif compression_method == "streaming_llm":
        print(f"Using StreamingLLM approach with {window_length} recent tokens and {max_length-window_length} attention sink tokens")
        # For StreamingLLM, we implement a simple version that keeps the first (max_length-window_length) tokens
        # and most recent window_length tokens
        from transformers.cache_utils import SinkCache
        kv_cache = SinkCache(
            window_length=window_length,
            num_sink_tokens=max_length-window_length
        )
    else:
        print("Using no compression (standard KV cache)")
        kv_cache = None
    
    return model, tokenizer, kv_cache

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

def evaluate_perplexity(args, model, tokenizer, method_name, kv_cache=None):
    """
    Evaluate perplexity on sequences from the dataset using token-by-token autoregressive generation.
    This approach mirrors real-world use of KV cache during generation.
    """
    print(f"Evaluating perplexity for method: {method_name}")
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    # Determine text field based on dataset
    text_field = "text" if "text" in dataset.column_names else dataset.column_names[0]
    
    # Setup tracking variables
    perplexity_by_position = {}
    
    # Define initial context length - how many tokens to process before starting measurements
    initial_context = 1024
    
    # Track number of samples processed
    num_processed = 0
    
    # Process sequences
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            if num_processed >= args.num_sequences:
                break
                
            # Get text from sample
            text = sample[text_field]
            if not isinstance(text, str) or len(text) < initial_context + 100:  # Skip very short texts
                continue
                
            # Encode the full text
            full_encoding = tokenizer(text, return_tensors="pt", truncation=True, 
                                      max_length=args.max_seq_length)
            full_input_ids = full_encoding["input_ids"].to(model.device)
            
            # Skip if too short after encoding
            if full_input_ids.size(1) < initial_context + 100:
                continue
                
            # Start with initial context
            input_ids = full_input_ids[:, :initial_context]
            
            # Initialize KV cache by running the model on initial context
            if kv_cache is not None:
                _ = model(input_ids=input_ids, past_key_values=kv_cache)
            else:
                # For baseline, we'll use the model's default KV cache handling
                past_key_values = None
                _ = model(input_ids=input_ids)
                
            print(f"Sample {num_processed+1}: Processing {full_input_ids.size(1) - initial_context} tokens auto-regressively")
            
            # Now process the rest token by token in an auto-regressive manner
            for pos in tqdm(range(initial_context, min(full_input_ids.size(1) - 1, args.max_seq_length - 1)),
                         desc="Token-by-token evaluation"):
                
                # Get current token (becomes the input)
                current_token = full_input_ids[:, :pos]
                
                # Get next token (ground truth label)
                next_token = full_input_ids[:, pos]
                
                # Forward pass with KV cache
                if kv_cache is not None:
                    outputs = model(input_ids=current_token, past_key_values=kv_cache)
                    kv_cache.key_cache=[]
                    kv_cache.value_cache=[]
                else:
                    outputs = model(input_ids=current_token, past_key_values=past_key_values)
                    # Update past_key_values for next iteration if not using custom cache
                    past_key_values = outputs.past_key_values
                
                # Get logits for current position
                logits = outputs.logits[:, -1, :]
                
                # Calculate log probability of the correct next token
                log_probs = torch.log_softmax(logits, dim=-1)
                correct_log_prob = log_probs[0, next_token]
                
                # Convert to perplexity (exp(-log_prob))
                token_perplexity = torch.exp(-correct_log_prob).item()
                
                # Store perplexity at this position
                if pos not in perplexity_by_position:
                    perplexity_by_position[pos] = []
                perplexity_by_position[pos].append(token_perplexity)
            
            # Increment processed count
            num_processed += 1
            
            # Free up memory between samples
            if kv_cache is None and past_key_values is not None:
                del past_key_values
                torch.cuda.empty_cache()
    
    # Calculate average perplexity by position
    avg_perplexity = {}
    for pos, values in perplexity_by_position.items():
        if values:  # Only calculate if we have values
            avg_perplexity[pos] = sum(values) / len(values)
    
    # Prepare results
    positions = sorted(avg_perplexity.keys())
    perplexities = [avg_perplexity[pos] for pos in positions]
    
    # Calculate final perplexity (average of last 100 tokens, or all if fewer)
    final_window = min(100, len(perplexities))
    final_perplexity = sum(perplexities[-final_window:]) / final_window if perplexities else float('nan')
    
    # Calculate overall average perplexity
    avg_overall = sum(perplexities) / len(perplexities) if perplexities else float('nan')
    
    results = {
        "method": method_name,
        "window_size": args.window_size,
        "topk": args.topk if hasattr(args, 'topk') else None,
        "positions": positions,
        "perplexities": perplexities,
        "average_perplexity": avg_overall,
        "final_perplexity": final_perplexity
    }
    
    # Save the results
    output_path = os.path.join(args.output_dir, f"{method_name}_perplexity.json")
    with open(output_path, "w") as f:
        # Convert values to JSON-serializable types
        serializable_results = {
            "method": results["method"],
            "window_size": results["window_size"],
            "topk": results["topk"],
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
    
    # Define colors and markers for different methods
    styles = {
        "none": {"color": "black", "marker": "o", "linestyle": "-", "label": "No Compression"},
        "q_filters": {"color": "blue", "marker": "s", "linestyle": "-", "label": "Q-Filters"},
        "k_norm": {"color": "green", "marker": "^", "linestyle": "-", "label": "K-norm"},
        "streaming_llm": {"color": "red", "marker": "x", "linestyle": "-", "label": "StreamingLLM"}
    }
    
    # Set up position bins for smoother plotting (average across neighboring positions)
    bin_size = 50  # Adjust based on your data density
    
    for results in results_list:
        method = results["method"]
        positions = results["positions"]
        perplexities = results["perplexities"]
        
        # Skip if no data
        if not positions or not perplexities:
            continue
            
        # Apply binning for smoother curves if we have many positions
        if len(positions) > 100:
            binned_positions = []
            binned_perplexities = []
            
            for i in range(0, len(positions), bin_size):
                end_idx = min(i + bin_size, len(positions))
                if end_idx > i:
                    bin_pos = sum(positions[i:end_idx]) / (end_idx - i)
                    bin_perp = sum(perplexities[i:end_idx]) / (end_idx - i)
                    binned_positions.append(bin_pos)
                    binned_perplexities.append(bin_perp)
                    
            positions = binned_positions
            perplexities = binned_perplexities
        
        # Get style for this method
        style = styles.get(method, {"color": "gray", "marker": ".", "linestyle": "-", "label": method})
        
        # Plot perplexity vs position with method-specific styling
        plt.plot(
            positions, 
            perplexities, 
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=f"{style['label']} (Window: {results['window_size']}, TopK: {results['topk']})",
            alpha=0.8,
            markevery=max(1, len(positions)//20)  # Show markers periodically
        )
    
    plt.xlabel("Position in Sequence")
    plt.ylabel("Perplexity")
    plt.title(f"Language Modeling Perplexity with Different KV Cache Compression Methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for final perplexity values
    for results in results_list:
        if results["positions"] and results["perplexities"]:
            final_pos = results["positions"][-1]
            final_perp = results["perplexities"][-1]
            plt.annotate(
                f"{final_perp:.2f}", 
                (final_pos, final_perp),
                textcoords="offset points",
                xytext=(5, 0),
                ha='left'
            )
    
    # Save the plot
    plot_path = os.path.join(args.output_dir, f"perplexity_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    
    # Create a second plot with logarithmic y-axis
    plt.figure(figsize=(10, 6))
    
    for results in results_list:
        method = results["method"]
        positions = results["positions"]
        perplexities = results["perplexities"]
        
        # Skip if no data
        if not positions or not perplexities:
            continue
            
        # Apply binning for smoother curves if we have many positions
        if len(positions) > 100:
            binned_positions = []
            binned_perplexities = []
            
            for i in range(0, len(positions), bin_size):
                end_idx = min(i + bin_size, len(positions))
                if end_idx > i:
                    bin_pos = sum(positions[i:end_idx]) / (end_idx - i)
                    bin_perp = sum(perplexities[i:end_idx]) / (end_idx - i)
                    binned_positions.append(bin_pos)
                    binned_perplexities.append(bin_perp)
                    
            positions = binned_positions
            perplexities = binned_perplexities
        
        # Get style for this method
        style = styles.get(method, {"color": "gray", "marker": ".", "linestyle": "-", "label": method})
        
        # Plot with log scale
        plt.semilogy(
            positions, 
            perplexities, 
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=f"{style['label']} (Window: {results['window_size']}, TopK: {results['topk']})",
            alpha=0.8,
            markevery=max(1, len(positions)//20)
        )
    
    plt.xlabel("Position in Sequence")
    plt.ylabel("Perplexity (log scale)")
    plt.title(f"Language Modeling Perplexity (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    # Save the log-scale plot
    log_plot_path = os.path.join(args.output_dir, f"perplexity_comparison_log.png")
    plt.savefig(log_plot_path, dpi=300, bbox_inches="tight")
    print(f"Log-scale plot saved to {log_plot_path}")
    
    # Create a bar chart comparing final perplexities
    plt.figure(figsize=(10, 6))
    
    # Extract method names and final perplexities
    methods = []
    final_perplexities = []
    method_colors = []
    
    for results in results_list:
        if "final_perplexity" in results and not np.isnan(results["final_perplexity"]):
            method_name = results["method"]
            style = styles.get(method_name, {"color": "gray", "label": method_name})
            
            # Create display name
            display_name = f"{style['label']} (W:{results['window_size']}, K:{results['topk']})"
            
            methods.append(display_name)
            final_perplexities.append(results["final_perplexity"])
            method_colors.append(style["color"])
    
    # Sort by perplexity
    if methods and final_perplexities:
        sorted_indices = np.argsort(final_perplexities)
        methods = [methods[i] for i in sorted_indices]
        final_perplexities = [final_perplexities[i] for i in sorted_indices]
        method_colors = [method_colors[i] for i in sorted_indices]
        
        # Create the bar chart
        bars = plt.barh(methods, final_perplexities, color=method_colors, alpha=0.7)
        
        # Add values as text
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                f"{final_perplexities[i]:.2f}",
                va='center'
            )
        
        plt.xlabel("Final Perplexity (last 100 tokens)")
        plt.title("Comparison of Final Perplexity Across Methods")
        plt.grid(True, alpha=0.3, axis='x')
        
        # Save the bar chart
        bar_plot_path = os.path.join(args.output_dir, f"final_perplexity_comparison.png")
        plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved to {bar_plot_path}")

def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store timestamp for uniquely identifying this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log file for output
    log_file = os.path.join(args.output_dir, f"perplexity_eval_{timestamp}.log")
    
    # Create a summary file
    with open(log_file, "w") as f:
        f.write("=== Language Modeling Perplexity Evaluation ===\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Window Size: {args.window_size}\n")
        f.write(f"TopK: {args.topk}\n")
        f.write(f"Max Sequence Length: {args.max_seq_length}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Number of Sequences: {args.num_sequences}\n")
        f.write(f"Output Directory: {args.output_dir}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
    
    # Run evaluations for specified compression methods
    results_list = []
    
    try:
        # Create a subfolder for this specific run
        run_dir = os.path.join(args.output_dir, f"ppl_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save a copy of the configuration
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        
        if args.use_q_filters:
            print("\n=== Evaluating Q-Filters ===")
            with open(log_file, "a") as f:
                f.write("\n=== Evaluating Q-Filters ===\n")
            
            model, tokenizer, kv_cache = setup_model(args, "q_filters")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "q_filters", kv_cache)
                results_list.append(results)
                
                # Save to the run subfolder
                with open(os.path.join(run_dir, "q_filters_results.json"), "w") as f:
                    json.dump(results, f, indent=2)
                
                # Log results
                with open(log_file, "a") as f:
                    f.write(f"Q-Filters - Avg Perplexity: {results['average_perplexity']:.4f}, Final: {results['final_perplexity']:.4f}\n")
                
            except Exception as e:
                print(f"Error during Q-Filters evaluation: {e}")
                with open(log_file, "a") as f:
                    f.write(f"Error during Q-Filters evaluation: {e}\n")
                import traceback
                traceback_str = traceback.format_exc()
                print(traceback_str)
                with open(log_file, "a") as f:
                    f.write(traceback_str + "\n")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        if args.use_k_norm:
            print("\n=== Evaluating K-norm ===")
            with open(log_file, "a") as f:
                f.write("\n=== Evaluating K-norm ===\n")
                
            model, tokenizer, kv_cache = setup_model(args, "k_norm")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "k_norm", kv_cache)
                results_list.append(results)
                
                # Save to the run subfolder
                with open(os.path.join(run_dir, "k_norm_results.json"), "w") as f:
                    json.dump(results, f, indent=2)
                
                # Log results
                with open(log_file, "a") as f:
                    f.write(f"K-norm - Avg Perplexity: {results['average_perplexity']:.4f}, Final: {results['final_perplexity']:.4f}\n")
                
            except Exception as e:
                print(f"Error during K-norm evaluation: {e}")
                with open(log_file, "a") as f:
                    f.write(f"Error during K-norm evaluation: {e}\n")
                import traceback
                traceback_str = traceback.format_exc()
                print(traceback_str)
                with open(log_file, "a") as f:
                    f.write(traceback_str + "\n")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        if args.use_streaming_llm:
            print("\n=== Evaluating StreamingLLM ===")
            with open(log_file, "a") as f:
                f.write("\n=== Evaluating StreamingLLM ===\n")
                
            model, tokenizer, kv_cache = setup_model(args, "streaming_llm")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "streaming_llm", kv_cache)
                results_list.append(results)
                
                # Save to the run subfolder
                with open(os.path.join(run_dir, "streaming_llm_results.json"), "w") as f:
                    json.dump(results, f, indent=2)
                
                # Log results
                with open(log_file, "a") as f:
                    f.write(f"StreamingLLM - Avg Perplexity: {results['average_perplexity']:.4f}, Final: {results['final_perplexity']:.4f}\n")
                
            except Exception as e:
                print(f"Error during StreamingLLM evaluation: {e}")
                with open(log_file, "a") as f:
                    f.write(f"Error during StreamingLLM evaluation: {e}\n")
                import traceback
                traceback_str = traceback.format_exc()
                print(traceback_str)
                with open(log_file, "a") as f:
                    f.write(traceback_str + "\n")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        # Run evaluation with no compression for baseline
        if not args.use_q_filters and not args.use_k_norm and not args.use_streaming_llm:
            args.use_baseline = True
        
        if hasattr(args, 'use_baseline') and args.use_baseline:
            print("\n=== Evaluating Baseline (No Compression) ===")
            with open(log_file, "a") as f:
                f.write("\n=== Evaluating Baseline (No Compression) ===\n")
                
            model, tokenizer, kv_cache = setup_model(args, "none")
            try:
                results = evaluate_perplexity(args, model, tokenizer, "none", kv_cache)
                results_list.append(results)
                
                # Save to the run subfolder
                with open(os.path.join(run_dir, "baseline_results.json"), "w") as f:
                    json.dump(results, f, indent=2)
                
                # Log results
                with open(log_file, "a") as f:
                    f.write(f"Baseline - Avg Perplexity: {results['average_perplexity']:.4f}, Final: {results['final_perplexity']:.4f}\n")
                
            except Exception as e:
                print(f"Error during baseline evaluation: {e}")
                with open(log_file, "a") as f:
                    f.write(f"Error during baseline evaluation: {e}\n")
                import traceback
                traceback_str = traceback.format_exc()
                print(traceback_str)
                with open(log_file, "a") as f:
                    f.write(traceback_str + "\n")
            finally:
                # Free up GPU memory
                del model
                torch.cuda.empty_cache()
        
        # Generate plots if requested and we have results to plot
        if args.plot and results_list:
            try:
                plot_results(results_list, args)
                
                # Copy plots to the run subfolder
                for plot_name in ["perplexity_comparison.png", "perplexity_comparison_log.png", "final_perplexity_comparison.png"]:
                    src_path = os.path.join(args.output_dir, plot_name)
                    if os.path.exists(src_path):
                        import shutil
                        dst_path = os.path.join(run_dir, plot_name)
                        shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error generating plots: {e}")
                with open(log_file, "a") as f:
                    f.write(f"Error generating plots: {e}\n")
        
        print("\nAll evaluations completed.")
        with open(log_file, "a") as f:
            f.write("\nAll evaluations completed.\n")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        with open(log_file, "a") as f:
            f.write("\nEvaluation interrupted by user.\n")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        with open(log_file, "a") as f:
            f.write(f"Unexpected error during evaluation: {e}\n")
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        with open(log_file, "a") as f:
            f.write(traceback_str + "\n")

if __name__ == "__main__":
    main() 