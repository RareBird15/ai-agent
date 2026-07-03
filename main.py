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


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")


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


def main() -> None:
    """Main function for the ai-agent.

    Raises:
        RuntimeError: If the OPENROUTER_API_KEY is not set in the environment variables.
    """
    parser = argparse.ArgumentParser(description="AI Agent Command Line Interface")
    parser.add_argument("user_prompt", type=str, help="User prompt for the AI agent")
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

    if response.usage is None:
        error_message = (
            "Response usage is None. Unable to retrieve token usage information."
        )
        raise RuntimeError(error_message)
    print(f"Prompt tokens: {response.usage.prompt_tokens}")  # noqa: T201
    print(f"Response tokens: {response.usage.completion_tokens}")  # noqa: T201
    print(response.choices[0].message.content)  # noqa: T201


if __name__ == "__main__":
    main()
