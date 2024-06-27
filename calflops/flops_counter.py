# !usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .calculate_pipeline import CalFlopsPipeline
from .utils import (
    flops_to_string,
    generate_transformer_input,
    macs_to_string,
    params_to_string,
)


def calculate_flops(
    model,
    input_shape=None,
    transformer_tokenizer=None,
    args=[],
    kwargs={},
    forward_mode="forward",
    include_backpropagation=False,
    compute_bp_factor=2.0,
    print_results=True,
    print_detailed=True,
    output_as_string=True,
    output_precision=2,
    output_unit=None,
    ignore_modules=None,
    is_sparse=False,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Args:
        model ([torch.nn.Module]): The model of input must be a PyTorch model.
        input_shape (tuple, optional): Input shape to the model. \
        If args|kwargs is empty, the model takes a tensor with this shape \
        as the only positional argument. \
        transformer_tokenizer (None, optional): Must be special if model type \
        is transformers and args|kwargs are empty.
        args (list, optional): list of positional arguments to the model, \
        such as bert input args is [input_ids, token_type_ids, attention_mask]. \
        Defaults to []
        kwargs (dict, optional): dictionary of keyword arguments to the model, \
        such as BERT input kwargs is
        {'input_ids': ..., 'token_type_ids':..., 'attention_mask':...}.
        Defaults to {}.
        forward_mode (str, optional): To determine the mode of model inference. \
        Defaults to 'forward'. Use 'generate' if model inference \
        uses model.generate().
        include_backpropagation (bool, optional): Decides whether the final
        FLOPs computation includes the computation for backpropagation. \
        compute_bp_factor (float, optional): The model's backpropagation \
        is a multiple of the forward propagation computation. Defaults to 2.
        print_results (bool, optional): Whether to print the model profile. \
        Defaults to True.
        print_detailed (bool, optional): Whether to print the detailed model profile. \
        Defaults to True.
        output_as_string (bool, optional): Whether to print the output as string. \
        Defaults to True.
        output_precision (int, optional) : Output holds the number of decimal places \
        if output_as_string is True. Default to 2.
        output_unit (str, optional): The unit used to output the result value,
        such as T, G, M, and K.
        Default is None, that is the unit of the output decide on value.
        ignore_modules ([type], optional): the list of modules to ignore \
        during profiling. Defaults to None.
        is_sparse (bool, optional): Whether to exclude sparse matrix flops. \
         Defaults to False.

    Returns:
        The number of floating-point operations, \
        multiply-accumulate operations (MACs), and parameters in the model.
    """

    assert isinstance(model, nn.Module), "model must be a PyTorch module"

    model.eval()

    is_transformer = True if "transformers" in str(type(model)) else False

    calculate_flops_pipeline = CalFlopsPipline(
        model=model,
        include_backpropagation=include_backpropagation,
        compute_bp_factor=compute_bp_factor,
        is_sparse=is_sparse,
    )
    calculate_flops_pipeline.start_flops_calculate(ignore_list=ignore_modules)

    device = next(model.parameters()).device
    model = model.to(device)

    if input_shape is not None:
        assert (
            len(args) == 0 and len(kwargs) == 0
        ), "args and kwargs must be empty value if \
        input_shape is not None, otherwise we generate random input of input_shape"
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"

        if transformer_tokenizer is None:  # model is not transformers model
            assert (
                is_transformer is False
            ), "the model is must not transformer model \
            if input_shape is not None and transformer_tokenizer is None"
            try:
                input = torch.ones(()).new_empty(
                    (*input_shape,),
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
            except StopIteration:
                input = torch.ones(()).new_empty((*input_shape,))
            args = [input]
        else:
            assert (
                len(input_shape) == 2
            ), "the format of input_shape must be (batch_size, seq_len) \
            if model is transformers model and auto_generate_transformers_input if True"
            kwargs = generate_transformer_input(
                input_shape=input_shape,
                model_tokenizer=transformer_tokenizer,
                device=device,
            )
    else:
        assert transformer_tokenizer or (
            len(args) > 0 or len(kwargs) > 0
        ), "one of input_shape, args or kwargs parameters must specified \
        if auto_generate_input is False"
        if transformer_tokenizer:
            kwargs = generate_transformer_input(
                input_shape=None, model_tokenizer=transformer_tokenizer, device=device
            )

    if kwargs:
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = value.to(device)
    else:
        kwargs = {}
        for index in range(len(args)):
            args[index] = args[index].to(device)

    if forward_mode == "forward":
        _ = model(*args, **kwargs)
    elif forward_mode == "generate":
        _ = model.generate(*args, **kwargs)
    else:
        raise NotImplementedError("forward_mode should be either forward or generate")

    flops = calculate_flops_pipeline.get_total_flops()
    macs = calculate_flops_pipeline.get_total_macs()
    params = calculate_flops_pipeline.get_total_params()

    if print_results:
        calculate_flops_pipeline.print_model_pipeline(
            units=output_unit, precision=output_precision, print_detailed=print_detailed
        )

    calculate_flops_pipeline.end_flops_calculate()

    if include_backpropagation:
        flops = flops * (1 + compute_bp_factor)
        macs = macs * (1 + compute_bp_factor)

    if output_as_string:
        return (
            flops_to_string(flops, units=output_unit, precision=output_precision),
            macs_to_string(macs, units=output_unit, precision=output_precision),
            params_to_string(params, units=output_unit, precision=output_precision),
        )

    return flops, macs, params
