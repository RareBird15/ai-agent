# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

import os
import subprocess

from openai.types.chat import ChatCompletionToolUnionParam


def run_python_file(
    working_directory: str,
    file_path: str,
    args: list[str] | None = None,
) -> str:
    """Run a Python file in a specified working directory with optional arguments.

    Args:
        working_directory (str): The directory in which to run the Python file.
        file_path (str): The path to the Python file to be executed.
        args (list[str] | None): Optional list of arguments to pass to the Python file.

    Returns:
        str: The output of the executed Python file.
    """
    try:  # noqa: PLW0717
        working_dir_abs: str = os.path.abspath(working_directory)
        target_file: str = os.path.normpath(os.path.join(working_dir_abs, file_path))

        valid_file_path: bool = (
            os.path.commonpath([working_dir_abs, target_file]) == working_dir_abs
        )
        if not valid_file_path:
            return (
                f'Error: Cannot execute "{file_path}" as it is outside the permitted '
                "working directory"
            )
        if not os.path.isfile(target_file):
            return f'Error: "{file_path}" does not exist or is not a regular file'
        if not target_file.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file'
        command = ["python", target_file]
        if args is not None:
            command.extend(args)
        completed_process = subprocess.run(
            command,
            cwd=working_dir_abs,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output_parts: list[str] = []
        if completed_process.returncode != 0:
            output_parts.append(
                f"Process exited with code {completed_process.returncode}",
            )
        if completed_process.stdout:
            output_parts.append(f"STDOUT:\n{completed_process.stdout}")
        if completed_process.stderr:
            output_parts.append(f"STDERR:\n{completed_process.stderr}")
        if not completed_process.stdout and not completed_process.stderr:
            output_parts.append("No output produced")
        return "\n".join(output_parts)
    except Exception as e:
        return f"Error: executing Python file: {e}"


schema_run_python_file: ChatCompletionToolUnionParam = {
    "type": "function",
    "function": {
        "name": "run_python_file",
        "description": (
            "Executes a Python file in a specified working directory with optional "
            "command-line arguments, returning the output or any errors"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "The path to the Python file to execute, relative to the "
                        "working directory"
                    ),
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of command-line arguments to pass to the Python "
                        "file"
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
}
