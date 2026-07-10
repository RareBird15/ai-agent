# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolUnionParam,
)

from functions.get_file_content import get_file_content, schema_get_file_content
from functions.get_files_info import get_files_info, schema_get_files_info
from functions.run_python_file import run_python_file, schema_run_python_file
from functions.write_file import schema_write_file, write_file

available_functions: list[ChatCompletionToolUnionParam] = [
    schema_get_file_content,
    schema_get_files_info,
    schema_run_python_file,
    schema_write_file,
]


def call_function(
    tool_call: ChatCompletionMessageToolCall,
    *,
    verbose: bool = False,
) -> dict[str, str]:
    """Call a function based on the tool call from the model.

    Args:
        tool_call (ChatCompletionMessageToolCall): The tool call from the model.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict[str, str]: The result of the function call.
    """
    function_name: str = tool_call.function.name
    function_args: dict[str, object] = json.loads(tool_call.function.arguments or "{}")
    if verbose:
        print(f" - Calling function: {function_name}({function_args})")  # noqa: T201
    else:
        print(f" - Calling function: {function_name}")  # noqa: T201
    function_map: dict[str, Callable[..., str]] = {
        "get_file_content": get_file_content,
        "get_files_info": get_files_info,
        "run_python_file": run_python_file,
        "write_file": write_file,
    }
    if function_name not in function_map:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Error: Unknown function: {function_name}",
        }
    function_args["working_directory"] = "./calculator"
    result: str = function_map[function_name](**function_args)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }
