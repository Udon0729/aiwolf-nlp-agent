"""Module defining the Pydantic AI action agent for vote/divine/guard/attack.

投票/占い/護衛/襲撃用の Pydantic AI アクションエージェントを定義するモジュール.
"""

from __future__ import annotations

import json

from pydantic_ai import Agent, RunContext

from llm.context import GameContext
from llm.models import ActionOutput
from llm.prompts import BASE_ACTION_PROMPT, ROLE_ACTION_PROMPTS, VOTE_PROMPT

action_agent = Agent(
    deps_type=GameContext,
    output_type=ActionOutput,
)


@action_agent.system_prompt
def base_prompt(_ctx: RunContext[GameContext]) -> str:
    """Provide base action rules as static system prompt.

    基本アクションルールを静的システムプロンプトとして提供する.
    """
    return BASE_ACTION_PROMPT


@action_agent.system_prompt
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
    return "\n".join(lines)


@action_agent.system_prompt
def strategy_and_emotional_context(ctx: RunContext[GameContext]) -> str:
    """Provide strategy memo, suspicion, and emotional/relationship state.

    戦略メモ・各プレイヤー評価・感情状態・関係性を提供する.
    """
    lines: list[str] = []
    if ctx.deps.strategy_memo:
        lines.append(f"=== 戦略メモ ===\n{ctx.deps.strategy_memo}")
    if ctx.deps.suspicion:
        lines.append("=== 各プレイヤー評価 ===")
        for name, eval_ in ctx.deps.suspicion.items():
            lines.append(f"  {name}: {eval_}")
    if ctx.deps.emotional_state:
        lines.append(f"=== 現在の感情状態 ===\n{ctx.deps.emotional_state}")
    if ctx.deps.relationships:
        lines.append("=== 他プレイヤーとの関係性 ===")
        for agent, rel in ctx.deps.relationships.items():
            lines.append(f"  {agent}: {rel}")
    return "\n".join(lines)


# === ツール群 ===


@action_agent.tool
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


@action_agent.tool
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


@action_agent.tool
def get_vote_results(ctx: RunContext[GameContext], day: int | None = None) -> str:
    """Get vote results.

    投票結果を取得する.

    Args:
        ctx: Run context with game state / ゲーム状態を含む実行コンテキスト
        day: Specific day to filter by, or None for all days / 特定の日を指定、Noneで全日分

    Returns:
        Formatted string of vote results / 投票結果のフォーマット文字列
    """
    votes = ctx.deps.vote_list
    if day is not None:
        votes = [v for v in votes if v.get("day") == day]
    if not votes:
        return "投票結果はありません。"
    lines = [f"Day{v['day']}: {v['agent']} → {v['target']}" for v in votes]
    return "投票結果:\n" + "\n".join(lines)


@action_agent.tool
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
    day_votes = [v for v in ctx.deps.vote_list if v.get("day") == day]

    lines = [f"=== {day}日目の要約 ==="]
    lines.append(f"発言数: {len(day_talks)}")
    if day_votes:
        vote_summary = json.dumps({str(v["agent"]): str(v["target"]) for v in day_votes}, ensure_ascii=False)
        lines.append(f"投票: {vote_summary}")
    if ctx.deps.executed_agent and ctx.deps.day == day + 1:
        lines.append(f"処刑: {ctx.deps.executed_agent}")
    if ctx.deps.attacked_agent and ctx.deps.day == day + 1:
        lines.append(f"襲撃: {ctx.deps.attacked_agent}")
    return "\n".join(lines)


def get_action_user_prompt(action_type: str, role: str) -> str:
    """Get the user prompt for a specific action type and role.

    アクションタイプと役職に応じたユーザープロンプトを取得する.

    Args:
        action_type: Type of action (vote/divine/guard/attack) / アクションタイプ
        role: Agent's role / エージェントの役職

    Returns:
        User prompt string / ユーザープロンプト文字列
    """
    if action_type == "vote":
        role_prompts = ROLE_ACTION_PROMPTS.get(role, {})
        return role_prompts.get("vote", VOTE_PROMPT)
    role_prompts = ROLE_ACTION_PROMPTS.get(role, {})
    return role_prompts.get(action_type, f"{action_type}の対象を選んでください。")
