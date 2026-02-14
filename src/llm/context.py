"""Module defining game context and LLM state for Pydantic AI integration.

Pydantic AI 統合用のゲームコンテキストと LLM 状態を定義するモジュール.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameContext:
    """Game state injected as Pydantic AI deps.

    Pydantic AI の deps として注入するゲーム状態.
    """

    agent_name: str
    role: str
    day: int
    alive_agents: list[str]
    talk_history: list[dict[str, str | int]]
    whisper_history: list[dict[str, str | int]]
    divine_results: list[dict[str, str | int]]
    medium_results: list[dict[str, str | int]]
    status_map: dict[str, str]
    role_map: dict[str, str]
    profile: str
    executed_agent: str | None
    attacked_agent: str | None
    vote_list: list[dict[str, str | int]]
    strategy_memo: str
    suspicion: dict[str, str]
    remain_count: int | None
    remain_length: int | None


@dataclass
class LLMState:
    """Internal LLM state maintained during a game. Reset on game start.

    ゲーム内で保持するLLMの内部状態. ゲーム開始時にリセット.
    """

    strategy_memo: str = ""
    suspicion: dict[str, str] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    talk_message_history: list[dict[str, str]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    whisper_message_history: list[dict[str, str]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    current_day: int = -1
    divine_results: list[dict[str, str | int]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    medium_results: list[dict[str, str | int]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    profile: str = ""
    vote_list: list[dict[str, str | int]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

    def reset_if_new_day(self, day: int) -> None:
        """Reset per-day state if the day has changed.

        日が変わった場合、日ごとの状態をリセットする.

        Args:
            day: Current day number / 現在の日数
        """
        if day != self.current_day:
            self.talk_message_history = []
            self.whisper_message_history = []
            self.current_day = day

    def reset_game(self) -> None:
        """Reset all state for a new game.

        ゲーム開始時に全状態をリセットする.
        """
        self.strategy_memo = ""
        self.suspicion = {}
        self.talk_message_history = []
        self.whisper_message_history = []
        self.current_day = -1
        self.divine_results = []
        self.medium_results = []
        self.profile = ""
        self.vote_list = []
