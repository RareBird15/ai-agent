# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

import os

from config import MAX_CHARS


def get_file_content(working_directory: str, file_path: str) -> str:
    """Get the content of a file.

    Args:
        working_directory: The working directory to resolve relative paths.
        file_path: The path to the file.

    Returns:
        The content of the file as a string.
    """
    try:  # noqa: PLW0717
        working_dir_abs: str = os.path.abspath(working_directory)
        target_file: str = os.path.normpath(os.path.join(working_dir_abs, file_path))

        valid_file_path: bool = (
            os.path.commonpath([working_dir_abs, target_file]) == working_dir_abs
        )
        if not valid_file_path:
            return (
                f'Error: Cannot read "{file_path}" as it is outside the permitted '
                "working directory"
            )
        if not os.path.isfile(target_file):
            return f'Error: File not found or is not a regular file: "{file_path}"'

        with open(target_file, encoding="utf-8") as file:
            content: str = file.read(MAX_CHARS)
            if file.read(1):
                content += (
                    f'[...File "{file_path}" truncated at {MAX_CHARS} characters]'
                )
        return content
    except (OSError, ValueError) as e:
        return f"Error: {e}"
