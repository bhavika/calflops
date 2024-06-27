# !usr/bin/env python
# -*- coding:utf-8 -*-


from calflops import calculate_flops
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

batch_size = 1
max_seq_length = 128
model_name = "llama2_hf_7B"
model_save = "../model/" + model_name
model = LlamaForCausalLM.from_pretrained(model_save)
tokenizer = LlamaTokenizer.from_pretrained(model_save)
flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, max_seq_length),
    transformer_tokenizer=tokenizer,
)
print("Llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
