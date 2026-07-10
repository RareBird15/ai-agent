# Copyright (C) 2026 RareBird15
"""Display/output formatting for the ai-agent package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent import handle_function_tool_calls

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


def print_response(
    response: ChatCompletion,
    user_prompt: str,
    *,
    verbose: bool,
) -> None:
    """Print the response from the OpenAI client.

    Args:
        response: The response from the OpenAI client.
        user_prompt: The original user prompt.
        verbose: A boolean indicating whether to print verbose output.

    Raises:
        RuntimeError: If verbose output is enabled and response usage is None.
    """
    message = response.choices[0].message

    if not verbose:
        if handle_function_tool_calls(message.tool_calls, verbose=verbose):
            return

        print(message.content)  # noqa: T201
        return

    usage = response.usage

    if usage is None:
        error_message = (
            "Response usage is None. Unable to retrieve token usage information."
        )
        raise RuntimeError(error_message)

    print(f"User prompt: {user_prompt}")  # noqa: T201
    print(f"Prompt tokens: {usage.prompt_tokens}")  # noqa: T201
    print(f"Response tokens: {usage.completion_tokens}")  # noqa: T201

    if handle_function_tool_calls(message.tool_calls, verbose=verbose):
        return

    print(message.content)  # noqa: T201
