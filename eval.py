import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import KNormCache
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from datasets import load_dataset
from lm_eval.tasks import get_task_dict

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

original_forward = type(model).forward
def patched_forward(self,*args,**kwargs):
    if hasattr(self, "_past_key_values") and self._past_key_values is not None:
        kwargs['past_key_values'] = self._past_key_values
    return original_forward(self, *args, **kwargs)
type(model).forward = patched_forward

model._past_key_values = KNormCache(
    window_length=64,
    max_length=128,
)

task_dict = get_task_dict(["longbench_narrativeqa"])
task = task_dict["longbench_narrativeqa"]
task.dataset['test'] = [ex for ex in task.dataset['test'] if ex['length'] < 8000]  # crude filter

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, trust_remote_code=True, max_gen_toks=128)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=[task],
    gen_kwargs={"max_length": None},
)

print(make_table(results))


