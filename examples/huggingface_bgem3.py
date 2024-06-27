# !usr/bin/env python
# -*- coding:utf-8 -*-

from calflops import calculate_flops_hf

batch_size = 1
max_seq_length = 128

model_name = "BAAI/bge-m3"

flops, macs, params, print_results = calculate_flops_hf(
    model_name=model_name,
    input_shape=(batch_size, max_seq_length),
    forward_mode="forward",
    print_results=False,
    return_results=True,
)

print(print_results)
print("%s FLOPs:%s  MACs:%s  Params:%s \n" % (model_name, flops, macs, params))
