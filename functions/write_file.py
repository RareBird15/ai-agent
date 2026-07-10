# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

import os

from openai.types.chat import ChatCompletionToolUnionParam


def write_file(working_directory: str, file_path: str, content: str) -> str:
    """Write content to a file.

    Args:
        working_directory (str): The working directory.
        file_path (str): The path to the file.
        content (str): The content to write.

    Returns:
        str: The path to the written file.
    """
    try:  # noqa: PLW0717
        working_dir_abs: str = os.path.abspath(working_directory)
        target_file: str = os.path.normpath(os.path.join(working_dir_abs, file_path))

        valid_file_path: bool = (
            os.path.commonpath([working_dir_abs, target_file]) == working_dir_abs
        )
        if not valid_file_path:
            return (
                f'Error: Cannot write to "{file_path}" as it is outside the permitted '
                "working directory"
            )
        if os.path.isdir(target_file):
            return f'Error: Cannot write to "{file_path}" as it is a directory'
        parent_dir = os.path.dirname(target_file)
        os.makedirs(parent_dir, exist_ok=True)

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)

        return (
            f'Successfully wrote to "{file_path}" ({len(content)} characters written)'
        )
    except (OSError, ValueError) as e:
        return f"Error: {e}"


schema_write_file: ChatCompletionToolUnionParam = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Writes content to a specified file relative to the working directory, "
            "creating any necessary parent directories"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "The path to the file to write, relative to the working "
                        "directory"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
    },
}
