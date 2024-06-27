# !usr/bin/env python
# -*- coding:utf-8 -*-

"""
This part of the code is inspired by ptflops and deepspeed profiling.
"""

from functools import partial

from .pytorch_ops import MODULE_HOOK_MAPPING
from .pytorch_ops import _patch_functionals
from .pytorch_ops import _patch_tensor_methods
from .pytorch_ops import _reload_functionals
from .pytorch_ops import _reload_tensor_methods
from .utils import flops_to_string
from .utils import get_module_flops
from .utils import get_module_macs
from .utils import macs_to_string
from .utils import number_to_string
from .utils import params_to_string
from .constants import DEFAULT_PRECISION

module_flop_count = []
module_mac_count = []
old_functions = {}


class CalFlopsPipline(object):
    """This pipeline calculates FLOPs and the
    number of parameters of each module in a PyTorch model.
    It is calculating the forward (and can optionally include back propagation) \
    pass for a PyTorch model and prints the model \
    graph with the calculated static attached to each module.
    It can return either just the FLOPs for a model \
    or also show where the FLOPs and parameters are used in the model,
    and which modules or layers could be a bottleneck in detail.
    """

    def __init__(self, model, include_backpropagation, compute_bp_factor, is_sparse):
        """Initial pipeline to calculate the FLOPs in a model.

        Args:
            model (pytorch model): This is a PyTorch model.
            include_backpropagation (bool): Whether to include backprop \
            in the calculation.
            compute_bp_factor (float): Defaults to 2.0. According to https://epochai.org/blog/backward-forward-FLOP-ratio
            is_sparse (bool): Whether to exclude sparse matrix flops.
        """

        self.model = model
        self.include_backpropagation = include_backpropagation
        self.compute_bp_factor = compute_bp_factor
        self.pipeline_started = False
        self.func_patched = False
        self.is_sparse = is_sparse  # Whether to exclude sparse matrix flops

    def start_flops_calculate(self, ignore_list=None):
        """Starts the pipeline for calculating FLOPs.

        Extra attributes are added recursively to all the \
        modules and torch.nn.functional is monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to \
            ignore while running the pipeline. Defaults to None.
        """

        self.reset_flops_calculate()
        _patch_functionals(old_functions, module_flop_count, module_mac_count)
        _patch_tensor_methods(old_functions, module_flop_count, module_mac_count)

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(
                        MODULE_HOOK_MAPPING[type(module)]
                    )
                return

            # if computing the flops of the functional in a module
            def pre_hook(module, input):
                module_flop_count.append([])
                module_mac_count.append([])

            if not hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                if module_flop_count:
                    module.__flops__ += sum([elem[1] for elem in module_flop_count[-1]])
                    module_flop_count.pop()
                    module.__macs__ += sum([elem[1] for elem in module_mac_count[-1]])
                    module_mac_count.pop()

            if not hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)

        self.model.apply(partial(register_module_hooks, ignore_list=ignore_list))
        self.pipeline_started = True
        self.func_patched = True

    def stop_flops_calculate(self):
        """Stop the pipeline calculating FLOPs.

        All torch.nn.functional are restored to their originals.
        """
        if self.pipeline_started and self.func_patched:
            _reload_functionals(old_functions)
            _reload_tensor_methods(old_functions)
            self.func_patched = False

        def remove_calculate_attrs(module):
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__

        self.model.apply(remove_calculate_attrs)

    def reset_flops_calculate(self):
        """Resets the pipeline calculating FLOPs.

        Adds or resets the extra attributes, include flops, macs, params.
        """

        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = (
                sum(
                    p.count_nonzero().item()
                    for p in module.parameters()
                    if p.requires_grad
                )
                if self.is_sparse
                else sum(p.numel() for p in module.parameters() if p.requires_grad)
            )

        self.model.apply(add_or_reset_attrs)

    def end_flops_calculate(self):
        """Ends the pipeline.
        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.pipeline_started:
            return
        self.stop_flops_calculate()
        self.pipeline_started = False

        def remove_calculate_attrs(module):
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__macs__"):
                del module.__macs__
            if hasattr(module, "__params__"):
                del module.__params__

        self.model.apply(remove_calculate_attrs)

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output FLOPs
            as a string. Defaults to False.

        Returns:
            The number of floating point operations of the model forward pass.
        """
        total_flops = get_module_flops(self.model, is_sparse=self.is_sparse)
        return number_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string=False):
        """Returns the total MACs of the model.

        Args:
            as_string (bool, optional): whether to output MACs as a string. \
            Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = get_module_macs(self.model, is_sparse=self.is_sparse)
        return macs_to_string(total_macs) if as_string else total_macs

    def get_total_params(self, as_string=False):
        """Returns the total number of parameters stored per rank.

        Args:
            as_string (bool, optional): whether to output parameters
            as a string. Defaults to False.

        Returns:
            The total number of parameters stored per rank.
        """
        total_params = self.model.__params__
        return params_to_string(total_params) if as_string else total_params

    def print_return_model_pipline(
        self,
        units=None,
        precision=DEFAULT_PRECISION,
        print_detailed=True,
        print_results=True,
    ):
        """Prints the model graph with the calculateing pipline attached to each module.

        Args:
            units: The units to use for the output. \
            precision (float, optional): The precision to use \
            for floating point output
            print_detailed (bool, optional): Whether to print a \
            detailed model profile.
            print_results (bool, optional): Whether to print results
        """
        if not self.pipeline_started:
            return

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_params = self.get_total_params()

        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        prints = []
        prints.append("\n----------- Calculate Flops Results ----------------")

        prints.append(
            "Notations:\n"
            + "number of parameters (Params), \
            number of multiply-accumulate operations(MACs),\n"
            + "number of floating-point operations (FLOPs), \
            floating-point operations per second (FLOPS),\n"
            + "fwd FLOPs (model forward propagation FLOPs), \
            bwd FLOPs (model backward propagation FLOPs),\n"
            + "default model backpropagation takes %.2f times as \
            much computation as forward propagation.\n"
            % self.compute_bp_factor
        )

        line_fmt = "{:<70}  {:<8}"
        prints.append(
            line_fmt.format("Total Training Params: ", params_to_string(total_params))
        )

        prints.append(
            line_fmt.format(
                "fwd MACs: ",
                macs_to_string(total_macs, units=units, precision=precision),
            )
        )
        prints.append(
            line_fmt.format(
                "fwd FLOPs: ",
                flops_to_string(total_flops, units=units, precision=precision),
            )
        )
        prints.append(
            line_fmt.format(
                "fwd+bwd MACs: ",
                macs_to_string(
                    total_macs * (1 + self.compute_bp_factor),
                    units=units,
                    precision=precision,
                ),
            )
        )
        prints.append(
            line_fmt.format(
                "fwd+bwd FLOPs: ",
                flops_to_string(
                    total_flops * (1 + self.compute_bp_factor),
                    units=units,
                    precision=precision,
                ),
            )
        )

        def flops_repr(module):
            params = module.__params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            items = [
                "{} = {:g}% Params".format(
                    params_to_string(params),
                    round(100 * params / total_params, precision)
                    if total_params
                    else 0,
                ),
                "{} = {:g}% MACs".format(
                    macs_to_string(macs),
                    round(100 * macs / total_macs, precision) if total_macs else 0,
                ),
                "{} = {:g}% FLOPs".format(
                    flops_to_string(flops),
                    round(100 * macs / total_flops, precision) if total_flops else 0,
                ),
            ]
            original_extra_repr = module.original_extra_repr()
            if original_extra_repr:
                items.append(original_extra_repr)
            return ", ".join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(add_extra_repr)

        if print_detailed:
            prints.append("\n--------------- Detailed FLOPs Results ----------------")
            prints.append(
                "Each module calculated is listed after its name "
                "in the following order: \n"
                "params, percentage of total params, MACs, "
                "percentage of total MACs, FLOPS, "
                "percentage of total FLOPs"
            )
            prints.append(
                "\nNote: 1. A module can have torch.nn.module or torch.nn.functional \
                to compute logits (e.g. CrossEntropyLoss). \
                \n They are not counted as submodules in \
                calflops and not to be printed out. \
                However they make up the difference between a parent's MACs \
                and the sum of its submodules'.\
                \n2. Number of floating-point operations is a theoretical estimation, \
                thus FLOPS computed using that could be larger than "
                "the maximum system throughput.\n"
            )
            prints.append(str(self.model))

        self.model.apply(del_extra_repr)

        prints.append(
            "---------------------------------------------------------------------------------------------------"
        )

        return_print = ""
        for line in prints:
            if print_results:
                print(line)
            return_print += line + "\n"
        return return_print

    def print_model_pipeline(
        self, units=None, precision=DEFAULT_PRECISION, print_detailed=True
    ):
        """Prints the model graph with the calculateing pipline attached to each module.

        Args:
            module_depth (int, optional): The depth of the model to \
            which to print the aggregated module information. When set to -1, \
            it prints information from the top to the \
            innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated \
            profile output to the number of top modules specified.
            print_detailed (bool, optional): Whether to print a \
            detailed model profile.
        """
        if not self.pipeline_started:
            return

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_params = self.get_total_params()

        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        print("\n-------------- Flops Results -------------")

        print(
            "Notations:\n"
            "number of parameters (Params), \
            number of multiply-accumulate operations(MACs),\n"
            "number of floating-point operations (FLOPs), \
            floating-point operations per second (FLOPS),\n"
            "fwd FLOPs (model forward propagation FLOPs), \
            bwd FLOPs (model backward propagation FLOPs),\n"
            "default model backpropagation takes %.2f times \
            as much computation as forward propagation.\n"
            % self.compute_bp_factor
        )

        line_fmt = "{:<70}  {:<8}"

        print(
            line_fmt.format("Total Training Params: ", params_to_string(total_params))
        )

        print(
            line_fmt.format(
                "fwd MACs: ",
                macs_to_string(total_macs, units=units, precision=precision),
            )
        )
        print(
            line_fmt.format(
                "fwd FLOPs: ",
                flops_to_string(total_flops, units=units, precision=precision),
            )
        )
        print(
            line_fmt.format(
                "fwd+bwd MACs: ",
                macs_to_string(
                    total_macs * (1 + self.compute_bp_factor),
                    units=units,
                    precision=precision,
                ),
            )
        )
        print(
            line_fmt.format(
                "fwd+bwd FLOPs: ",
                flops_to_string(
                    total_flops * (1 + self.compute_bp_factor),
                    units=units,
                    precision=precision,
                ),
            )
        )

        def flops_repr(module):
            params = module.__params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            items = [
                "{} = {:g}% Params".format(
                    params_to_string(params),
                    round(100 * params / total_params, precision)
                    if total_params
                    else 0,
                ),
                "{} = {:g}% MACs".format(
                    macs_to_string(macs),
                    round(100 * macs / total_macs, precision) if total_macs else 0,
                ),
                "{} = {:g}% FLOPs".format(
                    flops_to_string(flops),
                    round(100 * macs / total_flops, precision) if total_flops else 0,
                ),
            ]
            original_extra_repr = module.original_extra_repr()
            if original_extra_repr:
                items.append(original_extra_repr)
            return ", ".join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(add_extra_repr)

        if print_detailed:
            print("\n-------------- Detailed FLOPs Results ------------")
            print(
                "Each module calculated is listed after its name \
                in the following order: "
                "\nparams, percentage of total params, MACs, \
                percentage of total MACs, FLOPS, percentage of total FLOPs"
            )
            print(
                "\nNote: 1. A module can have torch.nn.module or \
                 torch.nn.functional to compute logits (e.g. CrossEntropyLoss). "
                "They are not counted as submodules in calflops \
                and not to be printed out. "
                "However they make up the difference between a \
                parent's MACs and the sum of its submodules'."
                "\n2. Number of floating-point operations is a theoretical estimation, "
                "thus FLOPS computed using that could be larger than the \
                maximum system throughput.\n"
            )
            print(self.model)

        self.model.apply(del_extra_repr)

        print(
            "---------------------------------------------------------------------------------------------------"
        )
