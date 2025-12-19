"""Module for Prompt Formatters used by LocalLLMClient.

LocalLLMClientで使用されるプロンプトフォーマッターのモジュール.
Strategyパターンにより、モデルごとのプロンプト形式の違いを吸収する.
"""

from abc import ABC, abstractmethod
from typing import Any

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class BasePromptFormatter(ABC):
    """Base class for prompt formatters.

    プロンプトフォーマッターの基底クラス.
    """

    @abstractmethod
    def format(
        self,
        system_prompt: str,
        user_prompt: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> str | list[int] | Any:
        """Format the prompts into model input.

        プロンプトをモデル入力形式に整形する.

        Args:
            system_prompt (str): System prompt / システムプロンプト
            user_prompt (str): User prompt / ユーザプロンプト
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer / トークナイザ

        Returns:
            str | list[int] | Any: Formatted prompt (text or tokens) / 整形されたプロンプト（テキストまたはトークン）
        """
        ...


class SimplePromptFormatter(BasePromptFormatter):
    """Formatter that simply concatenates prompts.

    プロンプトを単純に連結するフォーマッター.
    古い実装との互換性や、テンプレートを持たないモデル用.
    """

    def format(
        self,
        system_prompt: str,
        user_prompt: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, # noqa: ARG002
    ) -> str:
        """Format simply by concatenation.

        単純な連結による整形.

        Args:
            system_prompt (str): System prompt / システムプロンプト
            user_prompt (str): User prompt / ユーザプロンプト
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer (unused) / トークナイザ（未使用）

        Returns:
            str: Formatted text / 整形されたテキスト
        """
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


class ChatTemplateFormatter(BasePromptFormatter):
    """Formatter using tokenizer's chat template.

    トークナイザのチャットテンプレートを使用するフォーマッター.
    Hugging Face Transformersの推奨する方法.
    """

    def format(
        self,
        system_prompt: str,
        user_prompt: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> str:
        """Format using apply_chat_template.

        apply_chat_templateを使用した整形.

        Args:
            system_prompt (str): System prompt / システムプロンプト
            user_prompt (str): User prompt / ユーザプロンプト
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer / トークナイザ

        Returns:
            str: Formatted text / 整形されたテキスト
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # apply_chat_template returns str when tokenize=False
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ) # type: ignore[return-value]

