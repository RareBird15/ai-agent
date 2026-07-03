# Copyright (c) 2026 RareBird15
"""This file is part of ai-agent."""

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")


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

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": args.user_prompt,
            },
        ],
    )

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
