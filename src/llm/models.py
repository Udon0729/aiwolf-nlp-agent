"""Module defining structured output models for LLM responses.

LLM 応答用の構造化出力モデルを定義するモジュール.
"""

from __future__ import annotations

from pydantic import BaseModel


class TalkOutput(BaseModel):
    """Structured output for talk/whisper actions.

    トーク/囁きアクション用の構造化出力.
    """

    thought: str
    utterance: str
    strategy_memo: str
    suspicion: dict[str, str]


class ActionOutput(BaseModel):
    """Structured output for vote/divine/guard/attack actions.

    投票/占い/護衛/襲撃アクション用の構造化出力.
    """

    reason: str
    target: str
    strategy_memo: str
    suspicion: dict[str, str]
