# Copyright (C) 2026 RareBird15
"""This file is part of ai-agent."""

from openai.types.chat import ChatCompletionToolUnionParam

from functions.get_files_info import schema_get_files_info

available_functions: list[ChatCompletionToolUnionParam] = [
    schema_get_files_info,
]
