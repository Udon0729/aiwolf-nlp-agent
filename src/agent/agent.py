"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm.context import GameContext

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.comments: list[str] = []
        with Path.open(
            Path(str(self.config["path"]["random_talk"])),
            encoding="utf-8",
        ) as f:
            self.comments = f.read().splitlines()

        self.llm_enabled: bool = bool(config.get("llm", {}).get("enable", False))
        if self.llm_enabled:
            from llm.context import LLMState  # noqa: PLC0415

            self.llm_state = LLMState()

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
        self.agent_logger.logger.debug(packet)

        if self.llm_enabled:
            self._update_llm_state_from_packet()

    def _update_llm_state_from_packet(self) -> None:
        """Update LLM state from the current packet info.

        現在のパケット情報からLLM状態を更新する.
        """
        if not self.info:
            return
        if self.info.profile:
            self.llm_state.profile = self.info.profile
        if self.info.divine_result:
            dr = self.info.divine_result
            self.llm_state.divine_results.append(
                {
                    "day": dr.day,
                    "target": dr.target,
                    "result": str(dr.result.value),
                },
            )
        if self.info.medium_result:
            mr = self.info.medium_result
            self.llm_state.medium_results.append(
                {
                    "day": mr.day,
                    "target": mr.target,
                    "result": str(mr.result.value),
                },
            )
        if self.info.vote_list:
            self.llm_state.vote_list = [
                {"day": v.day, "agent": v.agent, "target": v.target} for v in self.info.vote_list
            ]

    def _build_context(self) -> GameContext:
        """Build GameContext from current game state.

        現在のゲーム状態から GameContext を構築する.

        Returns:
            GameContext: Built game context / 構築されたゲームコンテキスト
        """
        from llm.context import GameContext  # noqa: PLC0415  # runtime import for lazy loading

        talk_dicts: list[dict[str, str | int]] = [
            {"agent": t.agent, "day": t.day, "text": t.text, "turn": t.turn, "idx": t.idx}
            for t in self.talk_history
            if not t.skip and not t.over
        ]
        whisper_dicts: list[dict[str, str | int]] = [
            {"agent": t.agent, "day": t.day, "text": t.text, "turn": t.turn, "idx": t.idx}
            for t in self.whisper_history
            if not t.skip and not t.over
        ]
        status_map: dict[str, str] = {k: v.value for k, v in self.info.status_map.items()} if self.info else {}
        role_map: dict[str, str] = {k: v.value for k, v in self.info.role_map.items()} if self.info else {}
        return GameContext(
            agent_name=self.agent_name,
            role=self.role.value,
            day=self.info.day if self.info else 0,
            alive_agents=self.get_alive_agents(),
            talk_history=talk_dicts,
            whisper_history=whisper_dicts,
            divine_results=list(self.llm_state.divine_results),
            medium_results=list(self.llm_state.medium_results),
            status_map=status_map,
            role_map=role_map,
            profile=self.llm_state.profile,
            executed_agent=self.info.executed_agent if self.info else None,
            attacked_agent=self.info.attacked_agent if self.info else None,
            vote_list=list(self.llm_state.vote_list),
            strategy_memo=self.llm_state.strategy_memo,
            suspicion=dict(self.llm_state.suspicion),
            remain_count=self.info.remain_count if self.info else None,
            remain_length=self.info.remain_length if self.info else None,
        )

    def _llm_talk(self) -> str:
        """Call LLM talk agent for conversation.

        会話用のLLMトークエージェントを呼び出す.

        Returns:
            str: Generated utterance / 生成された発言
        """
        from llm.talk_agent import talk_agent  # noqa: PLC0415

        ctx = self._build_context()
        day = self.info.day if self.info else 0
        self.llm_state.reset_if_new_day(day)

        model = str(self.config["llm"]["model"])
        result = talk_agent.run_sync(
            "あなたの番です。発言してください。",
            deps=ctx,
            model=model,
            message_history=self.llm_state.talk_message_history,  # type: ignore[arg-type]
        )

        self.llm_state.talk_message_history = list(result.all_messages())  # type: ignore[assignment]
        self.llm_state.strategy_memo = result.output.strategy_memo
        self.llm_state.suspicion = dict(result.output.suspicion)

        self.agent_logger.logger.info("LLM thought: %s", result.output.thought)
        self.agent_logger.logger.info("LLM memo: %s", result.output.strategy_memo)

        return result.output.utterance

    def _llm_whisper(self) -> str:
        """Call LLM talk agent for whisper conversation.

        囁き会話用のLLMトークエージェントを呼び出す.

        Returns:
            str: Generated whisper message / 生成された囁きメッセージ
        """
        from llm.talk_agent import talk_agent  # noqa: PLC0415

        ctx = self._build_context()
        day = self.info.day if self.info else 0
        self.llm_state.reset_if_new_day(day)

        model = str(self.config["llm"]["model"])
        result = talk_agent.run_sync(
            "人狼同士の囁きです。仲間と戦略を相談してください。",
            deps=ctx,
            model=model,
            message_history=self.llm_state.whisper_message_history,  # type: ignore[arg-type]
        )

        self.llm_state.whisper_message_history = list(result.all_messages())  # type: ignore[assignment]
        self.llm_state.strategy_memo = result.output.strategy_memo
        self.llm_state.suspicion = dict(result.output.suspicion)

        self.agent_logger.logger.info("LLM whisper thought: %s", result.output.thought)

        return result.output.utterance

    def _llm_action(self, action_type: str) -> str:
        """Call LLM action agent for vote/divine/guard/attack.

        投票/占い/護衛/襲撃用のLLMアクションエージェントを呼び出す.

        Args:
            action_type: Type of action / アクションタイプ

        Returns:
            str: Target agent name / 対象エージェント名

        Raises:
            ValueError: If LLM returns an invalid target / LLMが無効なターゲットを返した場合
        """
        from llm.action_agent import action_agent, get_action_user_prompt  # noqa: PLC0415

        ctx = self._build_context()
        model = str(self.config["llm"]["model"])
        user_prompt = get_action_user_prompt(action_type, self.role.value)

        result = action_agent.run_sync(
            user_prompt,
            deps=ctx,
            model=model,
        )

        self.llm_state.strategy_memo = result.output.strategy_memo
        self.llm_state.suspicion = dict(result.output.suspicion)

        self.agent_logger.logger.info("LLM %s reason: %s", action_type, result.output.reason)
        self.agent_logger.logger.info("LLM %s target: %s", action_type, result.output.target)

        target = result.output.target
        if target not in self.get_alive_agents():
            self.agent_logger.logger.warning("LLMが無効なターゲットを返却: %s", target)
            raise ValueError(target)
        return target

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.llm_enabled:
            self.llm_state.reset_game()

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        if self.llm_enabled:
            try:
                return self._llm_whisper()
            except Exception:  # noqa: BLE001
                self.agent_logger.logger.warning("LLM失敗、ランダム発言にフォールバック", exc_info=True)
        return random.choice(self.comments)  # noqa: S311

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        if self.llm_enabled:
            try:
                return self._llm_talk()
            except Exception:  # noqa: BLE001
                self.agent_logger.logger.warning("LLM失敗、ランダム発言にフォールバック", exc_info=True)
        return random.choice(self.comments)  # noqa: S311

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        if self.llm_enabled:
            try:
                return self._llm_action("divine")
            except Exception:  # noqa: BLE001
                self.agent_logger.logger.warning("LLM失敗、ランダム選択にフォールバック", exc_info=True)
        return random.choice(self.get_alive_agents())  # noqa: S311

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        if self.llm_enabled:
            try:
                return self._llm_action("guard")
            except Exception:  # noqa: BLE001
                self.agent_logger.logger.warning("LLM失敗、ランダム選択にフォールバック", exc_info=True)
        return random.choice(self.get_alive_agents())  # noqa: S311

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        if self.llm_enabled:
            try:
                return self._llm_action("vote")
            except Exception:  # noqa: BLE001
                self.agent_logger.logger.warning("LLM失敗、ランダム選択にフォールバック", exc_info=True)
        return random.choice(self.get_alive_agents())  # noqa: S311

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        if self.llm_enabled:
            try:
                return self._llm_action("attack")
            except Exception:  # noqa: BLE001
                self.agent_logger.logger.warning("LLM失敗、ランダム選択にフォールバック", exc_info=True)
        return random.choice(self.get_alive_agents())  # noqa: S311

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
