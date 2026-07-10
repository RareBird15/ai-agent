# Copyright (c) 2026 RareBird15
"""This file is part of ai-agent."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from agent import generate_content
from display import print_response
from prompts import system_prompt

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


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
