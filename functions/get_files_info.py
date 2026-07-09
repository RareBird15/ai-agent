# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

import os

from openai.types.chat import ChatCompletionToolUnionParam


def get_files_info(working_directory: str, directory: str = ".") -> str:
    """Get information about files in a specified directory.

    Args:
        working_directory: The base working directory.
        directory: The target directory to check, relative to the working directory.

    Returns:
        A string indicating the result of the check.
    """
    try:  # noqa: PLW0717
        working_dir_abs: str = os.path.abspath(working_directory)
        target_dir: str = os.path.normpath(os.path.join(working_dir_abs, directory))

        valid_target_dir: bool = (
            os.path.commonpath([working_dir_abs, target_dir]) == working_dir_abs
        )
        if not valid_target_dir:
            return (
                f'Error: Cannot list "{directory}" '
                "as it is outside the permitted working directory"
            )

        if not os.path.isdir(target_dir):
            return f'Error: "{directory}" is not a directory'

        files_list: list[str] = []

        for item_name in os.listdir(target_dir):
            item_path: str = os.path.join(target_dir, item_name)
            file_size: int = os.path.getsize(item_path)
            is_dir: bool = os.path.isdir(item_path)
            files_list.append(
                f"- {item_name}: file_size={file_size} bytes, is_dir={is_dir}",
            )

    except (OSError, ValueError) as e:
        return f"Error: {e}"
    else:
        return "\n".join(files_list)


schema_get_files_info: ChatCompletionToolUnionParam = {
    "type": "function",
    "function": {
        "name": "get_files_info",
        "description": (
            "Lists files in a specified directory relative to the working "
            "directory, providing file size and directory status"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory path to list files from, relative to the "
                        "working directory (default is the working directory itself)"
                    ),
                },
            },
        },
    },
}
