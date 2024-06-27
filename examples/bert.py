# !usr/bin/env python
# -*- coding:utf-8 -*-

from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer

batch_size = 1
max_seq_length = 128

model_name = "hfl/chinese-roberta-wwm-ext/"
local_model_path = "../pretrained_models/" + model_name

model = AutoModel.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, max_seq_length),
    transformer_tokenizer=tokenizer,
)

print(f"{model_name} FLOPs:{flops} MACs:{macs} Params:{params} \n")
