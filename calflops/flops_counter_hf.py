# !usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn as nn
from transformers import AutoTokenizer

from .utils import (
    generate_transformer_input,
    flops_to_string,
    macs_to_string,
    params_to_string,
)
from .estimate import create_empty_model
from .calculate_pipeline import CalFlopsPipeline
from constants import DEFAULT_PRECISION, FORWARD_MODE, GENERATE_MODE, BACKPROP_FACTOR


def calculate_flops_hf(
    model_name,
    empty_model=None,
    input_shape=None,
    trust_remote_code=True,
    access_token=None,
    forward_mode=FORWARD_MODE,
    include_backpropagation=False,
    compute_bp_factor=BACKPROP_FACTOR,
    print_results=True,
    print_detailed=True,
    output_as_string=True,
    output_precision=DEFAULT_PRECISION,
    output_unit=None,
    ignore_modules=None,
    return_results=False,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Args:
        model_name (str): The model name on HuggingFace, \
        for example, meta-llama/Llama-2-7, baichuan-inc/Baichuan-13B-Chat etc.
        input_shape (tuple, optional): Input shape to the model. \
        If args|kwargs is empty, the model takes a tensor with \
        this shape as the only positional argument. Default to [].
        trust_remote_code (bool, optional): Required for custom models on HuggingFace.
        access_token (str, optional): HuggingFace access token for private/gated models.
        forward_mode (str, optional): To determine the mode of model inference,
        Defaults to 'forward'. Use 'generate' if model inference uses model.generate().
        include_backpropagation (bool, optional): Decides whether the final FLOPs \
        computation includes the computation for backpropagation.
        compute_bp_factor (float, optional): The model backpropagation is a \
        multiple of the forward propagation computation. Defaults to 2.
        print_results (bool, optional): Whether to print the model profile. \
        Defaults to True.
        print_detailed (bool, optional): Whether to print the detailed model profile. \
        Defaults to True.
        output_as_string (bool, optional): Whether to print the output as string. \
        Defaults to True.
        output_precision (int, optional) : Output holds the number of \
        decimal places if output_as_string is True. Default to 2.
        output_unit (str, optional): The unit used to output the result value, \
        such as T, G, M, and K. \
        Default is None, that is the unit of the output decide on value.
        ignore_modules ([type], optional): the list of modules to \
        ignore during profiling. Defaults to None.
        return_results (bool, optional): Whether to return the results. \
        Defaults to False.

    Returns:
        The number of floating-point operations, \
        multiply-accumulate operations (MACs), and parameters in the model.
    """

    if empty_model is None:
        empty_model = create_empty_model(
            model_name=model_name,
            library_name=None,
            trust_remote_code=trust_remote_code,
            access_token=access_token,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, access_token=access_token
    )

    assert isinstance(empty_model, nn.Module), "model must be a PyTorch module"
    device = next(empty_model.parameters()).device
    empty_model = empty_model.to(device)
    empty_model.eval()

    calculate_flops_pipeline = CalFlopsPipeline(
        model=empty_model,
        include_backpropagation=include_backpropagation,
        compute_bp_factor=compute_bp_factor,
        is_sparse=False,
    )

    calculate_flops_pipeline.start_flops_calculate(ignore_list=ignore_modules)

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        assert (
            len(input_shape) == 2
        ), "the format of input_shape must be (batch_size, seq_len) \
        if model is transformers model and auto_generate_transformers_input if True"
        kwargs = generate_transformer_input(
            input_shape=input_shape, model_tokenizer=tokenizer, device=device
        )
    else:
        kwargs = generate_transformer_input(
            input_shape=None, model_tokenizer=tokenizer, device=device
        )

    for key, value in kwargs.items():
        kwargs[key] = value.to(device)

    try:
        if forward_mode == FORWARD_MODE:
            _ = empty_model(**kwargs)
        if forward_mode == GENERATE_MODE:
            _ = empty_model.generate(**kwargs)
    except Exception as e:
        error_info = f"""The model:{model_name} encountered a problem in forwarding, 
        perhaps because the model cannot be deduced on meta device. 
        You can downloaded complete model parameters 
        locally from huggingface , \
        and then use the function:calflops.calculate_flops(model, tokenizer) \
        to calculate FLOPs on a GPU.\n
        Error details: {e}\n.
        """
        print(error_info)
        return None, None, None
    else:
        flops = calculate_flops_pipeline.get_total_flops()
        macs = calculate_flops_pipeline.get_total_macs()
        params = calculate_flops_pipeline.get_total_params()

        print_return = calculate_flops_pipeline.print_return_model_pipline(
            units=output_unit,
            precision=output_precision,
            print_detailed=print_detailed,
            print_results=print_results,
        )

        calculate_flops_pipeline.end_flops_calculate()

        if include_backpropagation:
            flops = flops * (1 + compute_bp_factor)
            macs = macs * (1 + compute_bp_factor)

        if output_as_string:
            flops = flops_to_string(
                flops, units=output_unit, precision=output_precision
            )
            macs = macs_to_string(macs, units=output_unit, precision=output_precision)
            params = params_to_string(
                params, units=output_unit, precision=output_precision
            )

        if return_results:
            return flops, macs, params, print_return
        else:
            return flops, macs, params
