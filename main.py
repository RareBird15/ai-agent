# Copyright (c) 2026 RareBird15
"""This file is part of ai-agent."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

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
    )


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
        RuntimeError: If the response usage is None.
    """
    if response.usage is None:
        error_message = (
            "Response usage is None. Unable to retrieve token usage information."
        )
        raise RuntimeError(error_message)

    if verbose:
        print(f"User prompt: {user_prompt}")  # noqa: T201
        print(f"Prompt tokens: {response.usage.prompt_tokens}")  # noqa: T201
        print(f"Response tokens: {response.usage.completion_tokens}")  # noqa: T201

    print(response.choices[0].message.content)  # noqa: T201


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
        {"role": "user", "content": args.user_prompt},
    ]
    response = generate_content(client, messages)

    print_response(response, args.user_prompt, verbose=args.verbose)


if __name__ == "__main__":
    main()
