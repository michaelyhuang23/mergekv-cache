# Q-Filters: Leveraging Query-Key Geometry for Efficient Key-Value Cache Compression
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789) 

![Q-Filters Demo GIF](assets/qfilters_demo.gif)

## Setup
1. Install required libraries in a virtual environment:
```bash
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
````
2. Configure HuggingFace\'s environment:
```bash
export HF_DATASETS_CACHE=<path_to_hf_cache>
export HF_HOME=<path_to_hf_cache>
export HF_TOKEN=<hf_token>
```

## Generate with Q-Filters
Here is an example of how to use Q-Filters in a generation setup:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import QFiltersCache
from datasets import load_dataset

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

question = """What is the probability of two integers selected at random having a greatest common divisor of 1."""
input_text = f"<|User|>{question}<|Assistant|><think>\n"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

past_key_values = QFiltersCache(
    window_length=64,
    max_length=128, 
    model_name=model_name
)

out = model.generate(
    **inputs,
    do_sample=True, 
    temperature=0.5, 
    max_new_tokens=4096, 
    past_key_values=past_key_values, 
    streamer=streamer
)
```

## Compute Q-Filters for a new model
1. Verify that the target model does not already have [pre-computed Q-Filters](https://huggingface.co/collections/nthngdy/q-filters-67a4994dcb302a3d37f3d119).
2. Use the `make_filters.py` script to generate the filters. For instance:
```bash
python make_filters.py \
--model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--model_cls Qwen2ForCausalLM \
--max_seq_len 2048 \
--num_sequences 10 \
--num_svd_samples 3000 \
--dataset_name PatrickHaller/fineweb-1B \
--save_mode disk \
# --save_mode hub \
# --save_mode hub+disk \
# --hf_user_id nthngdy \
--save_dir ../filters
```
3. For Q-Filters saved on disk, you can upload them later using this command:
```bash
huggingface-cli upload path_to_hf_repo path_to_local_qfilters .
```