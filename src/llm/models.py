"""Module defining structured output models for LLM responses.

LLM 応答用の構造化出力モデルを定義するモジュール.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TalkOutput(BaseModel):
    """Structured output for talk/whisper actions.

    トーク/囁きアクション用の構造化出力.
    """

    thought: str = Field(description="内部推論（戦略的思考過程。相手には見えない）")
    utterance: str = Field(description="実際に発言するテキスト（キャラクターの口調で自然に）")
    strategy_memo: str = Field(description="次回以降に引き継ぐ戦略メモ（現在の推理状況・方針の要約）")
    suspicion: dict[str, str] = Field(description="各プレイヤーへの評価（名前: 評価理由）")
    emotional_state: str = Field(description="現在の感情状態（例: 不安、疑念、怒り、安心、動揺）")
    relationships: dict[str, str] = Field(description="他プレイヤーとの関係性認識（名前: 関係性の説明）")


class ActionOutput(BaseModel):
    """Structured output for vote/divine/guard/attack actions.

    投票/占い/護衛/襲撃アクション用の構造化出力.
    """

    reason: str = Field(description="判断根拠（なぜこの対象を選んだか）")
    target: str = Field(description="対象エージェント名（生存者リストから選ぶこと）")
    strategy_memo: str = Field(description="次回以降に引き継ぐ戦略メモ")
    suspicion: dict[str, str] = Field(description="各プレイヤーへの評価（名前: 評価理由）")
    emotional_state: str = Field(description="現在の感情状態")
    relationships: dict[str, str] = Field(description="他プレイヤーとの関係性認識")
