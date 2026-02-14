"""Module defining the Pydantic AI talk agent for conversation actions.

会話アクション用の Pydantic AI トークエージェントを定義するモジュール.
"""

from __future__ import annotations

import json

from pydantic_ai import Agent, RunContext

from llm.context import GameContext
from llm.models import TalkOutput
from llm.prompts import BASE_TALK_PROMPT, NATURALNESS_GUIDELINES, ROLE_TALK_PROMPTS, WHISPER_PROMPT

talk_agent = Agent(
    deps_type=GameContext,
    output_type=TalkOutput,
)


@talk_agent.system_prompt
def base_prompt(_ctx: RunContext[GameContext]) -> str:
    """Provide base game rules as static system prompt.

    基本ゲームルールを静的システムプロンプトとして提供する.
    """
    return BASE_TALK_PROMPT


@talk_agent.system_prompt
def naturalness_prompt(_ctx: RunContext[GameContext]) -> str:
    """Provide naturalness guidelines as static system prompt.

    自然さガイドラインを静的システムプロンプトとして提供する.
    """
    return NATURALNESS_GUIDELINES


@talk_agent.system_prompt
def role_prompt(ctx: RunContext[GameContext]) -> str:
    """Provide role-specific guidelines as dynamic system prompt.

    役職別指針を動的システムプロンプトとして提供する.
    """
    return ROLE_TALK_PROMPTS.get(ctx.deps.role, "")


@talk_agent.system_prompt
def persona_prompt(ctx: RunContext[GameContext]) -> str:
    """Provide character persona from server-assigned profile.

    サーバから割り当てられたキャラクタープロフィールを人格層として提供する.
    """
    if not ctx.deps.profile:
        return ""
    return (
        f"=== あなたのキャラクター ===\n"
        f"{ctx.deps.profile}\n\n"
        "このキャラクターの性格・口調・年齢にふさわしい話し方をしてください。\n"
        "性格に基づく反応（例: 臆病なら不安を見せる、勇敢なら堂々とする）を自然に表現してください。\n"
        "ゲーム的に最適な発言よりも、このキャラクターが本当に言いそうなことを優先してください。"
    )


@talk_agent.system_prompt
def emotional_context(ctx: RunContext[GameContext]) -> str:
    """Provide emotional state and relationship context.

    感情状態と関係性コンテキストを提供する.
    """
    lines: list[str] = []

    if ctx.deps.emotional_state:
        lines.append(f"=== 現在の感情状態 ===\n{ctx.deps.emotional_state}")
        lines.append("この感情が発言のトーンや内容に自然に影響するようにしてください。")

    if ctx.deps.relationships:
        lines.append("=== 他プレイヤーとの関係性 ===")
        for agent, rel in ctx.deps.relationships.items():
            lines.append(f"  {agent}: {rel}")

    return "\n".join(lines)


@talk_agent.system_prompt
def game_state_prompt(ctx: RunContext[GameContext]) -> str:
    """Provide core game state as dynamic system prompt.

    コアゲーム状態を動的システムプロンプトとして提供する.
    """
    deps = ctx.deps
    lines = [
        "=== ゲーム状態 ===",
        f"名前: {deps.agent_name} / 役職: {deps.role} / {deps.day}日目",
        f"生存者: {', '.join(deps.alive_agents)}",
    ]
    if deps.divine_results:
        results = [f"  Day{r['day']}: {r['target']} → {r['result']}" for r in deps.divine_results]
        lines.append("占い結果:\n" + "\n".join(results))
    if deps.medium_results:
        results = [f"  Day{r['day']}: {r['target']} → {r['result']}" for r in deps.medium_results]
        lines.append("霊媒結果:\n" + "\n".join(results))
    if deps.executed_agent:
        lines.append(f"前日の処刑: {deps.executed_agent}")
    if deps.attacked_agent:
        lines.append(f"前夜の襲撃: {deps.attacked_agent}")
    if deps.remain_count is not None:
        lines.append(f"残り発言回数: {deps.remain_count}")
    if deps.remain_length is not None:
        lines.append(f"残り文字数: {deps.remain_length}")
    return "\n".join(lines)


@talk_agent.system_prompt
def strategy_context(ctx: RunContext[GameContext]) -> str:
    """Provide strategy memo and suspicion from previous turns.

    前回ターンの戦略メモと各プレイヤー評価を提供する.
    """
    lines: list[str] = []
    if ctx.deps.strategy_memo:
        lines.append(f"=== 戦略メモ（前回の自分の推論） ===\n{ctx.deps.strategy_memo}")
    if ctx.deps.suspicion:
        lines.append("=== 各プレイヤー評価 ===")
        for name, eval_ in ctx.deps.suspicion.items():
            lines.append(f"  {name}: {eval_}")
    return "\n".join(lines)


@talk_agent.system_prompt
def whisper_context(ctx: RunContext[GameContext]) -> str:
    """Provide whisper-specific context when in whisper mode.

    囁きモード時に囁き用コンテキストを提供する.
    """
    if ctx.deps.role == "WEREWOLF" and any(
        r == "WEREWOLF" for name, r in ctx.deps.role_map.items() if name != ctx.deps.agent_name
    ):
        allies = [name for name, r in ctx.deps.role_map.items() if r == "WEREWOLF" and name != ctx.deps.agent_name]
        return f"=== 人狼仲間 ===\n{', '.join(allies)}\n{WHISPER_PROMPT}"
    return ""


# === ツール群 ===


@talk_agent.tool
def search_talks_by_agent(ctx: RunContext[GameContext], agent_name: str) -> str:
    """Search all talks by a specific agent.

    特定プレイヤーの全発言を検索する.

    Args:
        ctx: Run context with game state / ゲーム状態を含む実行コンテキスト
        agent_name: Name of the agent to search / 検索対象のエージェント名

    Returns:
        Formatted string of matching talks / マッチした発言のフォーマット文字列
    """
    matches = [t for t in ctx.deps.talk_history if t.get("agent") == agent_name]
    if not matches:
        return f"{agent_name} の発言は見つかりませんでした。"
    lines = [f"Day{t['day']} Turn{t['turn']}: {t['text']}" for t in matches]
    return f"{agent_name} の発言:\n" + "\n".join(lines)


@talk_agent.tool
def search_talks_by_keyword(ctx: RunContext[GameContext], keyword: str) -> str:
    """Search talks containing a keyword.

    キーワードを含む発言を検索する.

    Args:
        ctx: Run context with game state / ゲーム状態を含む実行コンテキスト
        keyword: Keyword to search for / 検索キーワード

    Returns:
        Formatted string of matching talks / マッチした発言のフォーマット文字列
    """
    matches = [t for t in ctx.deps.talk_history if keyword in str(t.get("text", ""))]
    if not matches:
        return f"「{keyword}」を含む発言は見つかりませんでした。"
    lines = [f"Day{t['day']} {t['agent']}: {t['text']}" for t in matches]
    return f"「{keyword}」を含む発言:\n" + "\n".join(lines)


@talk_agent.tool
def get_my_past_talks(ctx: RunContext[GameContext], day: int | None = None) -> str:
    """Get the agent's own past talks.

    自分の過去の発言を取得する.

    Args:
        ctx: Run context with game state / ゲーム状態を含む実行コンテキスト
        day: Specific day to filter by, or None for all days / 特定の日を指定、Noneで全日分

    Returns:
        Formatted string of own past talks / 自分の過去の発言のフォーマット文字列
    """
    matches = [t for t in ctx.deps.talk_history if t.get("agent") == ctx.deps.agent_name]
    if day is not None:
        matches = [t for t in matches if t.get("day") == day]
    if not matches:
        return "過去の発言はありません。"
    lines = [f"Day{t['day']} Turn{t['turn']}: {t['text']}" for t in matches]
    return "自分の過去の発言:\n" + "\n".join(lines)


@talk_agent.tool
def get_day_summary(ctx: RunContext[GameContext], day: int) -> str:
    """Get a summary of events for a specific day.

    指定日のイベント要約を取得する.

    Args:
        ctx: Run context with game state / ゲーム状態を含む実行コンテキスト
        day: Day number to summarize / 要約する日数

    Returns:
        Formatted summary of the day's events / その日のイベント要約
    """
    day_talks = [t for t in ctx.deps.talk_history if t.get("day") == day]
    day_whispers = [t for t in ctx.deps.whisper_history if t.get("day") == day]
    day_votes = [v for v in ctx.deps.vote_list if v.get("day") == day]

    lines = [f"=== {day}日目の要約 ==="]
    lines.append(f"発言数: {len(day_talks)}")
    if day_whispers:
        lines.append(f"囁き数: {len(day_whispers)}")
    if day_votes:
        vote_summary = json.dumps({str(v["agent"]): str(v["target"]) for v in day_votes}, ensure_ascii=False)
        lines.append(f"投票: {vote_summary}")
    if ctx.deps.executed_agent and ctx.deps.day == day + 1:
        lines.append(f"処刑: {ctx.deps.executed_agent}")
    if ctx.deps.attacked_agent and ctx.deps.day == day + 1:
        lines.append(f"襲撃: {ctx.deps.attacked_agent}")
    return "\n".join(lines)
