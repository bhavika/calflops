# !usr/bin/env python
# -*- coding:utf-8 -*-

from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer

batch_size = 1
max_seq_length = 128
model_name = "hfl/chinese-roberta-wwm-ext/"
model_save = "../pretrain_models/" + model_name
model = AutoModel.from_pretrained(model_save)
tokenizer = AutoTokenizer.from_pretrained(model_save)

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, max_seq_length),
    transformer_tokenizer=tokenizer,
)
print(
    "Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n"
    % (flops, macs, params)
)
