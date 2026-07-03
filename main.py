# Copyright (c) 2026 RareBird15
"""This file is part of ai-agent."""

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
                "content": "Why is Boot.dev such a great place to learn backend development? Use one paragraph maximum.",  # noqa: E501
            },
        ],
    )

    print(response.choices[0].message.content)  # noqa: T201


if __name__ == "__main__":
    main()
