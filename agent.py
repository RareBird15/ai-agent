# Copyright (C) 2026 RareBird15
"""Agent logic for the ai-agent package."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from call_function import available_functions, call_function
from config import MODEL, TEMPERATURE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import OpenAI
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCall,
        ChatCompletionMessageToolCallUnion,
    )


def generate_content(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletion:
    """Generate content using the OpenAI client.

    Args:
        client: An instance of the OpenAI client.
        messages: A list of chat messages to send to the model.

    Returns:
        The generated content from the model.
    """
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=available_functions,
        temperature=TEMPERATURE,
    )


def handle_function_tool_calls(
    tool_calls: Sequence[ChatCompletionMessageToolCallUnion] | None,
    *,
    verbose: bool,
) -> bool:
    """Execute function tool calls if present.

    Args:
        tool_calls: Tool calls from the model message.
        verbose: Whether to print function results.

    Returns:
        True if one or more function tool calls were handled.

    Raises:
        RuntimeError: If a function call returns empty content.
    """
    if not tool_calls:
        return False

    handled_any = False

    for tool_call in tool_calls:
        if getattr(tool_call, "type", None) != "function":
            continue

        fn_tool_call = cast("ChatCompletionMessageToolCall", tool_call)
        result_message = call_function(fn_tool_call, verbose=verbose)

        if not result_message["content"]:
            error_message = "Function call returned empty content."
            raise RuntimeError(error_message)

        if verbose:
            print(f"-> {result_message['content']}")  # noqa: T201

        handled_any = True

    return handled_any
