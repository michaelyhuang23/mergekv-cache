import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import KNormCache, MergeKV
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from datasets import load_dataset
from lm_eval.tasks import get_task_dict
from src.attn_patch import patched_qwen2_attn_forward
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

Qwen2Attention.forward = patched_qwen2_attn_forward

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16",
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

orig_generate = model.generate
def patched_generate(*args, **kwargs):
    model._active_cache = MergeKV(compression_ratio=0.6, window_length=128, top_k=128)
    kwargs['past_key_values'] = model._active_cache
    kwargs['use_cache'] = True
    out = orig_generate(*args, **kwargs)
    del model._active_cache
    return out
model.generate = patched_generate


task = get_task_dict(['longbench_gov_report'])['longbench_gov_report']
print('total sample count', len(task.dataset['test']))
task.dataset['test'] = [ex for ex in task.dataset['test'] if ex['length'] < 8000 and ex['length'] > 256]
print('filtered sample count', len(task.dataset['test']))

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, trust_remote_code=True)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=[task],
    limit=1000,
    gen_kwargs={"max_length": None},
)

print(make_table(results))


