#!/usr/bin/env python

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import argparse

from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from accelerate import init_empty_weights
from accelerate.utils import (
    is_timm_available,
    is_transformers_available,
)


if is_transformers_available():
    import transformers
    from transformers import AutoConfig, AutoModel

if is_timm_available():
    import timm


def verify_on_hub(repo: str, token: str = None):
    """Verifies that the model is on the hub and returns model info."""
    try:
        return model_info(repo, token=token)
    except GatedRepoError:
        return "gated"
    except RepositoryNotFoundError:
        return "repo"


def check_has_model(error):
    """
    Checks which library spawned `error` when a model is not found
    """
    if (
        is_timm_available()
        and isinstance(error, RuntimeError)
        and "Unknown model" in error.args[0]
    ):
        return "timm"
    elif (
        is_transformers_available()
        and isinstance(error, OSError)
        and "does not appear to have a file named" in error.args[0]
    ):
        return "transformers"
    else:
        return "unknown"


def create_empty_model(
    model_name: str,
    library_name: str,
    trust_remote_code: bool = False,
    access_token: str = None,
):
    """
    Creates an empty model from its parent library on HF hub to
    calculate the overall memory consumption.

    Args:
        model_name (`str`):
            The model name on the Hub
        library_name (`str`):
            The library the model has an integration with, such as `transformers`.
            Will be used if `model_name` has no metadata to determine the library.
        trust_remote_code (`bool`, `optional`, defaults to `False`):
            Whether to execute the code present in the HF Hub for these models.
            This option should only be set to `True` for repositories you trust
            and in which you have read the code, as it will execute code present
            on the HF hub on your local machine.
        access_token (`str`, `optional`, defaults to `None`):
            HuggingFace access token to access private models.

    Returns:
        `torch.nn.Module`: A torch model that has been initialized on the `meta` device.

    """
    model_info = verify_on_hub(model_name, access_token)
    # Simplified errors
    if model_info == "gated":
        raise GatedRepoError(
            f"Repo for model `{model_name}` is gated. "
            f"You must be authenticated to access it. "
            f"Please run `huggingface-cli login`."
        )
    elif model_info == "repo":
        raise RepositoryNotFoundError(
            f"Repo for model `{model_name}` does not exist on the Hub. "
            f"If you are trying to access a private repo, "
            "make sure you are authenticated via `huggingface-cli login`."
        )
    if library_name is None:
        library_name = getattr(model_info, "library_name", False)
        if not library_name:
            raise ValueError(
                f"Model `{model_name}` does not have any library metadata on the Hub, "
                f"please manually pass in a `--library_name` to use."
            )
    if library_name == "transformers" or library_name == "sentence-transformers":
        if not is_transformers_available():
            raise ImportError(
                f"To check `{model_name}`, `transformers` must be installed. "
                f"Please install it via `pip install transformers`"
            )
        print(f"Loading pretrained config for `{model_name}` from `transformers`...")

        auto_map = model_info.config.get("auto_map", False)
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        with init_empty_weights():
            # remote code could specify a specific `AutoModel` class in the `auto_map`
            constructor = AutoModel
            if isinstance(auto_map, dict):
                value = None
                for key in auto_map.keys():
                    if key.startswith("AutoModelFor"):
                        value = key
                        break
                if value is not None:
                    constructor = getattr(transformers, value)
            model = constructor.from_config(config, trust_remote_code=trust_remote_code)
    elif library_name == "timm":
        if not is_timm_available():
            raise ImportError(
                f"To use `{model_name}`, `timm` must be installed. "
                f"Please install it via `pip install timm`"
            )
        print(f"Loading pretrained config for `{model_name}` from `timm`...")
        with init_empty_weights():
            model = timm.create_model(model_name, pretrained=False)
    else:
        raise ValueError(
            f"Library `{library_name}` is not supported yet."
        )
    return model


def create_ascii_table(headers: list, rows: list, title: str):
    """Creates a pretty table from a list of rows, minimal version of `tabulate`."""
    sep_char, in_between = "│", "─"
    column_widths = []
    for i in range(len(headers)):
        column_values = [row[i] for row in rows] + [headers[i]]
        max_column_width = max(len(value) for value in column_values)
        column_widths.append(max_column_width)

    formats = [f"%{column_widths[i]}s" for i in range(len(rows[0]))]

    pattern = f"{sep_char}{sep_char.join(formats)}{sep_char}"
    diff = 0

    def make_row(left_char, middle_char, right_char):
        middle_section = middle_char.join([in_between * n for n in column_widths])
        return f"{left_char}{middle_section}{in_between * diff}{right_char}"

    separator = make_row("├", "┼", "┤")
    if len(title) > sum(column_widths):
        diff = abs(len(title) - len(separator))
        column_widths[-1] += diff

    # Update with diff
    separator = make_row("├", "┼", "┤")
    initial_rows = [
        make_row("┌", in_between, "┐"),
        f"{sep_char}{title.center(len(separator) - 2)}{sep_char}",
        make_row("├", "┬", "┤"),
    ]
    table = "\n".join(initial_rows) + "\n"
    column_widths[-1] += diff
    centered_line = [text.center(column_widths[i]) for i, text in enumerate(headers)]
    table += f"{pattern % tuple(centered_line)}\n{separator}\n"
    for i, line in enumerate(rows):
        centered_line = [t.center(column_widths[i]) for i, t in enumerate(line)]
        table += f"{pattern % tuple(centered_line)}\n"
    table += f'└{"┴".join([in_between * n for n in column_widths])}┘'

    return table
