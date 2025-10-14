import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import KNormCache, TopKKV, MergeKV
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from datasets import load_dataset
from lm_eval.tasks import get_task_dict
from src.attn_patch import patched_qwen2_attn_forward
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

Qwen2Attention.forward = patched_qwen2_attn_forward

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16",
    #attn_implementation="flash_attention_2"
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

orig_generate = model.generate
def patched_generate(*args, **kwargs):
    model._active_cache = MergeKV(compression_ratio=0.8, window_ratio=0.0256, top_k_ratio=0.0256)
    #model._active_cache = TopKKV(compression_ratio=0.5, window_length=128, top_k=128)
    #model._active_cache = KNormCache(compression_ratio=0.8, window_length=128)
    kwargs['past_key_values'] = model._active_cache
    kwargs['use_cache'] = True
    out = orig_generate(*args, **kwargs)
    del model._active_cache
    return out
model.generate = patched_generate


task = get_task_dict(['longbench_gov_report'])['longbench_gov_report']
print('total sample count', len(task.dataset['test']))
task.dataset['test'] = [ex for ex in task.dataset['test'] if ex['length'] < 5000 and ex['length'] > 256]
print('filtered sample count', len(task.dataset['test']))

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=3, trust_remote_code=True)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=[task],
    num_fewshot=0,
    gen_kwargs={"max_length": None, "max_new_tokens": 256, "do_sample": False,     # Greedy = faster, consistent
        "temperature": 0,
    },
)

print(make_table(results))


