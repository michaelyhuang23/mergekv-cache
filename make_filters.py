from tqdm import tqdm
import pickle
import os
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

import modeling as M
from src.utils import QFilters


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--model_name")
parser.add_argument("--model_cls")
parser.add_argument("--max_seq_len", type=int, default=2048)
parser.add_argument("--num_sequences", type=int, default=20)
parser.add_argument("--num_svd_samples", type=int, default=3000)
parser.add_argument("--filter_suffix", default="")
parser.add_argument("--torch_dtype", default="bfloat16")

parser.add_argument("--dataset_name")
parser.add_argument("--dataset_config", default="default")
parser.add_argument("--dataset_split", default="train[:1000]")

parser.add_argument("--save_mode", default="disk")
parser.add_argument("--save_dir", default="")
parser.add_argument("--hf_user_id", default="")


args = parser.parse_args()

model_name = args.model_name
model_cls = getattr(M, args.model_cls)
max_seq_len = args.max_seq_len
num_sequences = args.num_sequences
num_svd_samples = args.num_svd_samples
filter_suffix = args.filter_suffix
torch_dtype = args.torch_dtype

dataset_name = args.dataset_name
dataset_config = args.dataset_config
dataset_split = args.dataset_split

save_mode = args.save_mode
save_dir = args.save_dir
hf_user_id = args.hf_user_id

if "disk" in save_mode and not save_dir:
    raise ValueError("In 'disk' or 'disk+hub' save modes, a '--save_dir' must be provided.")

if "hub" in save_mode and not hf_user_id:
    raise ValueError("In 'hub' or 'disk+hub' save modes, a '--hf_user_id' must be provided.")



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model_cls.from_pretrained(
    model_name, attn_implementation="flash_attention_2", device_map="auto",  low_cpu_mem_usage=True, torch_dtype=torch_dtype)

model = model.eval()

dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)


with torch.no_grad():
    decoder = getattr(model, "gpt_neox", getattr(model, 'model', None))
    svd_filters = [[] for _ in range(len(decoder.layers))]
    sample_count = 0
    num_k_heads = None

    for i, sample in tqdm(enumerate(dataset)):

        tokens = tokenizer(sample["text"], return_tensors="pt")
        if tokens.input_ids.shape[-1] < max_seq_len:
            continue
        sample_count+=1
        input_ids = tokens.input_ids[:, :max_seq_len].to(device)
        if sample_count < num_sequences:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                out_repr = model(input_ids).past_key_values
            for j, (query, key) in enumerate(out_repr): # TODO These are KV not QK!!
                num_k_heads = key.shape[1]
                svd_filters[j].append(query.flatten(0, 1).cpu())
        else:
            break

    del model

    for f_id, el in enumerate(svd_filters):
        stacked_el = torch.stack(el, 1).flatten(1, 2)
        idx = torch.argsort(torch.rand(stacked_el.shape[1], device=stacked_el.device))[:num_svd_samples]
        stacked_el = stacked_el[:, idx].cuda()
        u,s,vh = torch.linalg.svd(stacked_el.float())
        svd_sign = ((u[..., 0]>0).float().mean(-1) > 0.5).float()*2-1
        svd_filter_q = -svd_sign[:, None] * vh[..., 0, :]
        svd_filters[f_id] = svd_filter_q.reshape(num_k_heads, -1, svd_filter_q.shape[-1]).mean(-2)

    svd_filters = torch.nn.Parameter(torch.stack(svd_filters))
    q_filters = QFilters(*svd_filters.shape)
    q_filters.q_filters = svd_filters

    model_suffix = model_name.split("/")[-1]
    filter_savename = f"{model_suffix}_qfilt{'_' + filter_suffix if filter_suffix else ''}"
    if "disk" in save_mode:
        q_filters.save_pretrained(f"{save_dir}/{filter_savename}")
    if "hub" in save_mode:
        q_filters.push_to_hub(f"{hf_user_id}/{filter_savename}")
