"""Module defining the base class for LLM clients.

LLMクライアントの基底クラスを定義するモジュール.
"""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for LLM clients.

    LLMクライアントの基底クラス.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text based on prompts.

        プロンプトに基づいてテキストを生成する.

        Args:
            system_prompt (str): System prompt / システムプロンプト
            user_prompt (str): User prompt / ユーザプロンプト

        Returns:
            str: Generated text / 生成されたテキスト
        """
        ...

