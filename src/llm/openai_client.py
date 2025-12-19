"""Module for OpenAI LLM client.

OpenAI LLMクライアントのモジュール.
"""

import os
from typing import Any

from openai import OpenAI

from llm.client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API.

    OpenAI API用クライアント.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the OpenAI client.

        OpenAIクライアントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # For simplicity, we might want to log a warning or raise an error
            # But here we just proceed, the client might fail later if key is missing
            pass
            
        self.client = OpenAI(api_key=api_key)
        self.model = str(config.get("model", "gpt-4o"))
        self.temperature = float(config.get("temperature", 0.7))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using OpenAI API.

        OpenAI APIを使用してテキストを生成する.

        Args:
            system_prompt (str): System prompt / システムプロンプト
            user_prompt (str): User prompt / ユーザプロンプト

        Returns:
            str: Generated text / 生成されたテキスト
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        return content if content else ""

