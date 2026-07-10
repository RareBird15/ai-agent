# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

from openai.types.chat import ChatCompletionToolUnionParam

from functions.get_file_content import schema_get_file_content
from functions.get_files_info import schema_get_files_info
from functions.run_python_file import schema_run_python_file
from functions.write_file import schema_write_file


available_functions: list[ChatCompletionToolUnionParam] = [
    schema_get_file_content,
    schema_get_files_info,
    schema_run_python_file,
    schema_write_file,
]
