from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch


BASE_MODEL = "decapoda-research/llama-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, torch_dtype=torch.float16,)
model = PeftModel.from_pretrained(model, "./checkpoint").to("cuda")