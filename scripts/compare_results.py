#!/usr/bin/env python3
# Compare results from different compression methods

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RULER benchmark results across compression methods")
    parser.add_argument("--result_dirs", nargs="+", required=True, help="Directories containing result files")
    parser.add_argument("--output_dir", type=str, default="comparisons", help="Directory to save comparisons")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    return parser.parse_args()


def load_results(result_dirs):
    """
    Load results from specified directories.
    """
    all_results = {}
    
    for result_dir in result_dirs:
        result_path = Path(result_dir) / "results.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                results = json.load(f)
                
            # Extract method and compression ratio from directory name
            dir_name = Path(result_dir).name
            parts = dir_name.split("_")
            
            if len(parts) >= 2 and parts[1].endswith("x"):
                method = parts[0]
                compression_ratio = parts[1]
                
                # Use both method and compression ratio as key
                key = f"{method}_{compression_ratio}"
                all_results[key] = results
            else:
                # If naming doesn't follow convention, use directory name
                all_results[dir_name] = results
    
    return all_results


def extract_metrics(all_results):
    """
    Extract relevant metrics from results for comparison.
    """
    metrics = {}
    
    for method, results in all_results.items():
        # Extract the overall average score
        if "results" in results and "ruler" in results["results"]:
            ruler_results = results["results"]["ruler"]
            
            # Create metric dictionary for this method
            metrics[method] = {
                "average": ruler_results.get("average", 0),
            }
            
            # Extract individual task scores if available
            for task, score in ruler_results.items():
                if task != "average" and not isinstance(score, dict):
                    metrics[method][task] = score
    
    return metrics


def create_comparison_table(metrics):
    """
    Create a pandas DataFrame for comparing results.
    """
    # Convert metrics dictionary to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Sort columns to have 'average' first, then alphabetical
    columns = ["average"] + sorted([col for col in df.columns if col != "average"])
    df = df[columns]
    
    return df


def plot_comparisons(df, output_dir):
    """
    Generate comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot average scores
    plt.figure(figsize=(10, 6))
    df["average"].sort_values(ascending=False).plot(kind="bar", color="skyblue")
    plt.title("Average RULER Score by Compression Method")
    plt.ylabel("Score")
    plt.xlabel("Compression Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "average_scores.png")
    
    # Plot individual task scores
    plt.figure(figsize=(12, 8))
    task_cols = [col for col in df.columns if col != "average"]
    df[task_cols].plot(kind="bar", figsize=(12, 8))
    plt.title("RULER Task Scores by Compression Method")
    plt.ylabel("Score")
    plt.xlabel("Compression Method")
    plt.xticks(rotation=45)
    plt.legend(title="Task", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "task_scores.png")
    
    # Plot comparison heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(df[task_cols].values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(label="Score")
    plt.xticks(range(len(task_cols)), task_cols, rotation=45, ha="right")
    plt.yticks(range(len(df)), df.index)
    plt.title("RULER Task Performance Heatmap")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "performance_heatmap.png")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    all_results = load_results(args.result_dirs)
    
    if not all_results:
        print("No valid result files found in the specified directories.")
        return
    
    # Extract metrics
    metrics = extract_metrics(all_results)
    
    # Create comparison table
    comparison_df = create_comparison_table(metrics)
    
    # Save comparison table
    csv_path = Path(args.output_dir) / "comparison.csv"
    comparison_df.to_csv(csv_path)
    print(f"Comparison table saved to {csv_path}")
    
    # Display comparison table
    print("\nComparison of RULER benchmark results:")
    print(comparison_df)
    
    # Generate plots if requested
    if args.plot:
        plot_comparisons(comparison_df, args.output_dir)
        print(f"Comparison plots saved to {args.output_dir}")


if __name__ == "__main__":
    main() 