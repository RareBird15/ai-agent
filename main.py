# Copyright (c) 2026 RareBird15
"""This file is part of ai-agent."""

from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from openai import OpenAI

from call_function import available_functions
from prompts import system_prompt

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


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
        model="openrouter/free",
        messages=messages,
        tools=available_functions,
        temperature=0,
    )


def _print_function_tool_calls(tool_calls: list[Any] | None) -> bool:
    """Print function tool calls if present.

    Args:
        tool_calls: Tool calls from the model message.

    Returns:
        True if one or more function tool calls were printed, False otherwise.
    """
    if not tool_calls:
        return False

    printed_any = False

    for tool_call in tool_calls:
        if getattr(tool_call, "type", None) != "function":
            continue

        function_obj = getattr(tool_call, "function", None)
        if function_obj is None:
            continue

        function_name_raw = getattr(function_obj, "name", "unknown")
        function_name: str = (
            function_name_raw if isinstance(function_name_raw, str) else "unknown"
        )

        arguments_raw = getattr(function_obj, "arguments", "{}")
        arguments_json: str = arguments_raw if isinstance(arguments_raw, str) else "{}"

        try:
            function_args: dict[str, object] = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            function_args = {}

        print(f"Calling function: {function_name}({function_args})")  # noqa: T201
        printed_any = True

    return printed_any


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
        if _print_function_tool_calls(message.tool_calls):
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

    if _print_function_tool_calls(message.tool_calls):
        return

    print(message.content)  # noqa: T201


def main() -> None:
    """Main function for the ai-agent.

    Raises:
        RuntimeError: If the OPENROUTER_API_KEY is not set in the environment variables.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    parser = argparse.ArgumentParser(description="AI Agent Command Line Interface")
    parser.add_argument("user_prompt", type=str, help="User prompt for the AI agent")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if not api_key:
        error_message = "OPENROUTER_API_KEY is not set in the environment variables."
        raise RuntimeError(error_message)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    response = generate_content(client, messages)

    print_response(response, args.user_prompt, verbose=args.verbose)


if __name__ == "__main__":
    main()
