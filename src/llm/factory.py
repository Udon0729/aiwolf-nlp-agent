"""Module for LLM client factory.

LLMクライアントファクトリのモジュール.
"""

from typing import Any

from llm.client import BaseLLMClient
from llm.openai_client import OpenAIClient


class LLMFactory:
    """Factory class for creating LLM clients.

    LLMクライアントを作成するファクトリクラス.
    """

    @staticmethod
    def create(config: dict[str, Any]) -> BaseLLMClient | None:
        """Create an LLM client based on configuration.

        設定に基づいてLLMクライアントを作成する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書

        Returns:
            BaseLLMClient | None: Created LLM client or None if not configured / 作成されたLLMクライアント、または設定されていない場合はNone
        """
        llm_config = config.get("llm")
        if not llm_config:
            return None

        llm_type = llm_config.get("type", "openai")

        if llm_type == "openai":
            return OpenAIClient(llm_config)
        
        if llm_type == "local":
            try:
                from llm.local_client import LocalLLMClient
                return LocalLLMClient(llm_config)
            except ImportError:
                # Handle case where dependencies are missing
                return None
            except Exception:
                # Handle loading errors
                return None

        return None

