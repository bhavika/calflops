from calflops import calculate_flops
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

batch_size = 1
max_seq_length = 128

model_name = "llama2_hf_7B"
local_model_location = "../model/" + model_name

model = LlamaForCausalLM.from_pretrained(local_model_location)
tokenizer = LlamaTokenizer.from_pretrained(local_model_location)

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, max_seq_length),
    transformer_tokenizer=tokenizer,
)

print(f"Llama2(7B) FLOPs:{flops}   MACs:{macs}   Params:{params}")
