import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import islice
from matplotlib.colors import LogNorm

def generate_attention_plots(model_name, text_samples, max_len=128):
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"  # Added to handle attention implementation warning
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Setup hooks for Q, K, V extraction
    num_layers = model.config.num_hidden_layers
    q_collect = [[] for _ in range(num_layers)]
    k_collect = [[] for _ in range(num_layers)]
    v_collect = [[] for _ in range(num_layers)]

    def make_hook(collect_list):
        def hook(module, input, output):
            collect_list.append(output.detach())
        return hook

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.register_forward_hook(make_hook(q_collect[i]))
        layer.self_attn.k_proj.register_forward_hook(make_hook(k_collect[i]))
        layer.self_attn.v_proj.register_forward_hook(make_hook(v_collect[i]))

    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads

    qq_sums, kk_sums, vv_sums = {}, {}, {}
    num_samples = len(text_samples)
    print(f"Processing {num_samples} text samples on {device}...")

    for i, text_sample in enumerate(text_samples):
        print(f"  Aggregating sample {i + 1}/{num_samples}")
        inputs = tokenizer(text_sample, return_tensors="pt", max_length=max_len, padding='max_length', truncation=True).to(device)
        
        # Compute position_ids and cos, sin
        position_ids = torch.arange(0, inputs["input_ids"].shape[-1], dtype=torch.long, device=device).unsqueeze(0)
        seq_len = position_ids.shape[1]
        dummy_x = torch.zeros(1, seq_len, model.config.hidden_size, dtype=torch.bfloat16, device=device)
        cos, sin = model.model.rotary_emb(dummy_x, position_ids)
        
        # Clear collections before forward
        for cl in q_collect + k_collect + v_collect:
            cl.clear()
        
        with torch.no_grad():
            _ = model(**inputs)

        for layer_idx in range(num_layers):
            Q_pre = q_collect[layer_idx][0].squeeze(0)
            K_pre = k_collect[layer_idx][0].squeeze(0)
            V_pre = v_collect[layer_idx][0].squeeze(0)
            
            # Reshape and apply RoPE to Q and K
            query_states = Q_pre.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
            key_states = K_pre.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # Reshape back
            Q_post = query_states.transpose(1, 2).contiguous().view(1, seq_len, -1).squeeze(0)
            K_post = key_states.transpose(1, 2).contiguous().view(1, seq_len, -1).squeeze(0)
            V_post = V_pre

            # To numpy
            Q_np = Q_post.cpu().to(torch.float32).numpy()
            K_np = K_post.cpu().to(torch.float32).numpy()
            V_np = V_post.cpu().to(torch.float32).numpy()
            
            qq = np.dot(Q_np, Q_np.T) / np.linalg.norm(Q_np, axis=-1)[:, None] / np.linalg.norm(Q_np, axis=-1)[None, :]
            kk = np.dot(K_np, K_np.T) / np.linalg.norm(K_np, axis=-1)[:, None] / np.linalg.norm(K_np, axis=-1)[None, :]
            vv = np.dot(V_np, V_np.T) / np.linalg.norm(V_np, axis=-1)[:, None] / np.linalg.norm(V_np, axis=-1)[None, :]

            # Initialize and accumulate sums
            qq_sums[layer_idx] = qq_sums.get(layer_idx, np.zeros_like(qq)) + qq
            kk_sums[layer_idx] = kk_sums.get(layer_idx, np.zeros_like(kk)) + kk
            vv_sums[layer_idx] = vv_sums.get(layer_idx, np.zeros_like(vv)) + vv

    base_plot_dir = os.path.join("plots", model_name.replace("/", "_"))
    print(f"\nSaving aggregated plots to: {base_plot_dir}")
    os.makedirs(base_plot_dir, exist_ok=True)

    for layer_idx in qq_sums:
        # Calculate averages
        avg_qq = qq_sums[layer_idx] / num_samples
        avg_kk = kk_sums[layer_idx] / num_samples
        avg_vv = vv_sums[layer_idx] / num_samples
        
        # Take absolute value and add small epsilon for log scale
        abs_kk = np.abs(avg_kk) + 1e-8
        abs_qq = np.abs(avg_qq) + 1e-8
        abs_vv = np.abs(avg_vv) + 1e-8
        
        # Find shared vmin and vmax
        vmin = min(np.min(abs_kk), np.min(abs_qq), np.min(abs_vv))
        vmax = max(np.max(abs_kk), np.max(abs_qq), np.max(abs_vv))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.heatmap(abs_kk, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), ax=axes[0])
        axes[0].set_title(f"Key-Key (Layer {layer_idx+1})")
        axes[0].invert_yaxis()
        
        sns.heatmap(abs_qq, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), ax=axes[1])
        axes[1].set_title(f"Query-Query (Layer {layer_idx+1})")
        axes[1].invert_yaxis()
        
        sns.heatmap(abs_vv, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), ax=axes[2])
        axes[2].set_title(f"Value-Value (Layer {layer_idx+1})")
        axes[2].invert_yaxis()
        
        plt.suptitle(f"Layer {layer_idx+1} Aggregated Dot Products")
        plt.tight_layout()
        save_path = os.path.join(base_plot_dir, f"layer_{layer_idx+1}.png")
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    model_to_analyze = "Qwen/Qwen2.5-1.5B"
    NUM_SAMPLES_TO_AGGREGATE = 50
    MAX_TOKEN_LENGTH = 512
    
    print("Attempting to load wikitext dataset sample...")
    text_samples = ["The quick brown fox jumps over the lazy dog."] # Fallback
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
        # Filter for non-empty lines and take the specified number of samples
        filtered_dataset = (line['text'] for line in dataset if line['text'].strip())
        text_samples = list(islice(filtered_dataset, NUM_SAMPLES_TO_AGGREGATE))
        if not text_samples:
            raise ValueError("Dataset is empty or could not be loaded properly.")
    except Exception as e:
        print(f"Could not load wikitext, using a default sentence. Error: {e}")

    print(f"\nAggregating over {len(text_samples)} samples.\n")
    generate_attention_plots(
        model_name=model_to_analyze, 
        text_samples=text_samples, 
        max_len=MAX_TOKEN_LENGTH
    )
    print("\nAnalysis complete. Check the 'plots' directory for the output images.")