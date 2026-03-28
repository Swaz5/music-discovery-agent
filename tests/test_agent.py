"""
Tests for MusicDiscoveryAgent.

All Anthropic API calls and tool executions are fully mocked.
No real network calls are made in any test.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.discovery_agent import MusicDiscoveryAgent, _MAX_ITERATIONS


# ── Mock-response builders ─────────────────────────────────────────────────────


def _usage(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    return u


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_block(name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = call_id
    b.name = name
    b.input = args
    return b


def _response(stop_reason: str, content: list, *, tokens=(100, 50)) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    r.usage = _usage(*tokens)
    return r


# ── Shared fixture text ────────────────────────────────────────────────────────

_FINAL_TEXT = """\
**Portishead — Glory Box**
Artist: Portishead
Track: Glory Box
Genre/tags: trip-hop, electronic, dark
Why: Haunting vocals over downtempo beats with deep bass presence. Energy 0.42.
Vibe description: Like wandering through a foggy harbour at 3 AM.
Vibe match: 0.92
Deezer: https://www.deezer.com/track/123456
"""


def _make_agent(**kwargs) -> MusicDiscoveryAgent:
    """Instantiate an agent with the Anthropic client patched out."""
    with patch("src.agent.discovery_agent.anthropic.AsyncAnthropic"):
        return MusicDiscoveryAgent(**kwargs)


# ── Loop termination ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_terminates_on_end_turn():
    """Agent returns after the first end_turn response."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=_response("end_turn", [_text_block(_FINAL_TEXT)])
    )
    with patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("dark atmospheric music")

    assert result["iterations"] == 1
    assert mock_client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_agent_stops_at_max_iterations():
    """Loop exits after _MAX_ITERATIONS even without end_turn."""
    always_tool = _response("tool_use", [_tool_block("search_by_tag", {"tag": "jazz"})])
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=always_tool)

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", AsyncMock(return_value=[])),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("jazz music")

    assert result["iterations"] == _MAX_ITERATIONS
    assert mock_client.messages.create.call_count == _MAX_ITERATIONS
    assert isinstance(result["recommendations"], list)


@pytest.mark.asyncio
async def test_agent_unknown_stop_reason_breaks_loop():
    """Unexpected stop_reason exits cleanly without raising."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_response("max_tokens", []))

    with patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("query")

    assert result["iterations"] == 1
    assert isinstance(result["recommendations"], list)


@pytest.mark.asyncio
async def test_agent_completes_after_tool_then_end_turn():
    """Agent runs: tool call → tool result → end_turn correctly."""
    tool_resp = _response("tool_use", [_tool_block("search_by_tag", {"tag": "ambient"})])
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", AsyncMock(return_value=[])),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("ambient music")

    assert result["iterations"] == 2
    assert len(result["recommendations"]) == 1


# ── Token accumulation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_total_tokens_sum_across_iterations():
    """total_tokens accumulates input+output across every API call."""
    tool_resp = _response("tool_use", [_tool_block("search_by_tag", {"tag": "jazz"})],
                          tokens=(200, 80))
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)], tokens=(300, 100))

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", AsyncMock(return_value=[])),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("query")

    assert result["total_tokens"] == (200 + 80) + (300 + 100)


# ── Tool dispatch ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_dispatches_correct_tool_name_and_args():
    """execute_tool is called with exactly the name and arguments Claude specified."""
    tool_resp = _response(
        "tool_use",
        [_tool_block("search_by_tag", {"tag": "shoegaze", "limit": 5}, call_id="x")],
    )
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])
    mock_execute = AsyncMock(return_value=[{"name": "My Bloody Valentine"}])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", mock_execute),
    ):
        agent = MusicDiscoveryAgent()
        await agent.discover("shoegaze music")

    mock_execute.assert_called_once_with("search_by_tag", {"tag": "shoegaze", "limit": 5})


@pytest.mark.asyncio
async def test_agent_handles_multiple_tool_calls_in_one_turn():
    """Agent processes two parallel tool_use blocks in a single response."""
    tool_resp = _response(
        "tool_use",
        [
            _tool_block("search_by_tag", {"tag": "ambient"}, "c1"),
            _tool_block("search_by_tag", {"tag": "drone"}, "c2"),
        ],
    )
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])
    mock_execute = AsyncMock(return_value=[])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", mock_execute),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("atmospheric music")

    assert mock_execute.call_count == 2
    assert len(result["reasoning_trace"]) == 2


# ── Reasoning trace ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reasoning_trace_empty_when_no_tools_called():
    """No tool calls → empty reasoning_trace."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=_response("end_turn", [_text_block(_FINAL_TEXT)])
    )
    with patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("query")

    assert result["reasoning_trace"] == []


@pytest.mark.asyncio
async def test_reasoning_trace_records_tool_name_and_args():
    """Successful tool calls appear in the trace with tool name and arguments."""
    tool_resp = _response("tool_use", [_tool_block("search_artist", {"name": "Boards of Canada"})])
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool",
              AsyncMock(return_value={"name": "Boards of Canada"})),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("idm music")

    trace = result["reasoning_trace"]
    assert len(trace) == 1
    assert trace[0]["tool"] == "search_artist"
    assert trace[0]["arguments"] == {"name": "Boards of Canada"}
    assert "result_preview" in trace[0]


@pytest.mark.asyncio
async def test_reasoning_trace_records_iteration_number():
    """Each trace entry captures the iteration it came from."""
    r1 = _response("tool_use", [_tool_block("search_knowledge_base", {"query": "jazz"}, "c1")])
    r2 = _response("tool_use", [_tool_block("search_by_tag", {"tag": "bebop"}, "c2")])
    r3 = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[r1, r2, r3])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", AsyncMock(return_value="ctx")),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("jazz")

    trace = result["reasoning_trace"]
    assert trace[0]["iteration"] == 1
    assert trace[1]["iteration"] == 2


# ── Error handling ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_error_captured_in_trace():
    """Tool execution errors land in reasoning_trace['error'] and loop continues."""
    tool_resp = _response(
        "tool_use",
        [_tool_block("get_audio_features", {"track_name": "X", "artist_name": "Y"})],
    )
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool",
              AsyncMock(side_effect=RuntimeError("Deezer 429"))),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("music query")

    trace = result["reasoning_trace"]
    assert len(trace) == 1
    assert "error" in trace[0]
    assert "Deezer 429" in trace[0]["error"]


@pytest.mark.asyncio
async def test_tool_error_does_not_abort_loop():
    """After a tool error Claude receives an error result and the loop continues."""
    tool_resp = _response("tool_use", [_tool_block("search_artist", {"name": "Unknown"})])
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool",
              AsyncMock(side_effect=KeyError("not found"))),
    ):
        agent = MusicDiscoveryAgent()
        result = await agent.discover("jazz")

    assert result["iterations"] == 2
    assert len(result["recommendations"]) > 0


@pytest.mark.asyncio
async def test_tool_error_message_sent_back_to_claude():
    """The tool_result block for a failed tool carries an error message."""
    tool_resp = _response("tool_use", [_tool_block("search_by_tag", {"tag": "jazz"}, "err1")])
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool",
              AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        agent = MusicDiscoveryAgent()
        await agent.discover("jazz")

    # Second call's messages list should include a tool_result with is_error=True
    second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
    # Last user message should be the error tool_result
    last_user = second_call_messages[-1]
    assert last_user["role"] == "user"
    assert last_user["content"][0]["is_error"] is True


# ── System prompt & budget awareness ──────────────────────────────────────────


def test_build_system_prompt_contains_remaining_budget():
    """Budget placeholder is filled with the remaining iteration count."""
    agent = _make_agent()

    prompt_start = agent._build_system_prompt(iterations_used=0)
    assert str(_MAX_ITERATIONS) in prompt_start

    prompt_mid = agent._build_system_prompt(iterations_used=5)
    assert str(_MAX_ITERATIONS - 5) in prompt_mid


def test_build_system_prompt_budget_never_goes_negative():
    """Budget is clamped to 0 even if iterations_used exceeds the limit."""
    agent = _make_agent()
    prompt = agent._build_system_prompt(iterations_used=_MAX_ITERATIONS + 5)
    assert "0" in prompt


def test_build_system_prompt_appends_taste_context_when_present():
    """Preference context is appended when the engine returns non-empty context."""
    mock_prefs = MagicMock()
    mock_prefs.get_recommendation_context.return_value = "## User taste profile\ntrip-hop"

    agent = _make_agent(preference_engine=mock_prefs)
    prompt = agent._build_system_prompt()

    assert "## User taste profile" in prompt
    assert "trip-hop" in prompt


def test_build_system_prompt_no_context_appended_when_empty():
    """Empty recommendation context leaves the base prompt unchanged."""
    mock_prefs = MagicMock()
    mock_prefs.get_recommendation_context.return_value = ""

    agent = _make_agent(preference_engine=mock_prefs)
    prompt = agent._build_system_prompt()

    assert "## User taste profile" not in prompt


@pytest.mark.asyncio
async def test_system_prompt_passed_to_anthropic_on_each_call():
    """The system prompt is passed as the 'system' kwarg on every API call."""
    tool_resp = _response("tool_use", [_tool_block("search_by_tag", {"tag": "jazz"})])
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", AsyncMock(return_value=[])),
    ):
        agent = MusicDiscoveryAgent()
        await agent.discover("music")

    for call in mock_client.messages.create.call_args_list:
        assert "system" in call[1]
        assert len(call[1]["system"]) > 100


# ── _parse_recommendations ─────────────────────────────────────────────────────


def test_parse_empty_text_returns_empty_list():
    agent = _make_agent()
    assert agent._parse_recommendations("") == []
    assert agent._parse_recommendations("   \n  ") == []


def test_parse_single_recommendation_all_fields():
    agent = _make_agent()
    recs = agent._parse_recommendations(_FINAL_TEXT)

    assert len(recs) == 1
    r = recs[0]
    assert r["artist"] == "Portishead"
    assert r["track"] == "Glory Box"
    assert "trip-hop" in r["genre_tags"]
    assert "electronic" in r["genre_tags"]
    assert r["vibe_match_score"] == 0.92
    assert r["deezer_url"] == "https://www.deezer.com/track/123456"
    assert "foggy harbour" in r["vibe_description"]
    assert "Energy 0.42" in r["why"]


def test_parse_multiple_recommendations():
    agent = _make_agent()
    text = """\
**Portishead — Glory Box**
Artist: Portishead
Track: Glory Box
Genre/tags: trip-hop
Why: Haunting.
Vibe description: Foggy.
Vibe match: 0.92
Deezer: N/A

**Massive Attack — Teardrop**
Artist: Massive Attack
Track: Teardrop
Genre/tags: trip-hop, ambient
Why: Ethereal.
Vibe description: Floating.
Vibe match: 0.88
Deezer: https://www.deezer.com/track/999
"""
    recs = agent._parse_recommendations(text)
    assert len(recs) == 2
    assert recs[0]["artist"] == "Portishead"
    assert recs[1]["artist"] == "Massive Attack"
    assert recs[1]["deezer_url"] == "https://www.deezer.com/track/999"


def test_parse_genre_tags_split_by_comma():
    agent = _make_agent()
    text = """\
**X — Y**
Artist: X
Track: Y
Genre/tags: trip-hop, electronic, dark
Why: Good.
Vibe description: Nice.
Vibe match: 0.8
Deezer: N/A
"""
    recs = agent._parse_recommendations(text)
    assert set(recs[0]["genre_tags"]) == {"trip-hop", "electronic", "dark"}


def test_parse_vibe_score_normalised_from_integer():
    """A raw score of 8 (not 0.8) is normalised to 0.8."""
    agent = _make_agent()
    text = """\
**X — Y**
Artist: X
Track: Y
Genre/tags: rock
Why: Good.
Vibe description: Nice.
Vibe match: 8
Deezer: N/A
"""
    recs = agent._parse_recommendations(text)
    assert recs[0]["vibe_match_score"] == pytest.approx(0.8)


def test_parse_no_deezer_url_when_na():
    agent = _make_agent()
    text = """\
**X — Y**
Artist: X
Track: Y
Genre/tags: rock
Why: Good.
Vibe description: Nice.
Vibe match: 0.75
Deezer: N/A
"""
    recs = agent._parse_recommendations(text)
    assert recs[0]["deezer_url"] == ""


def test_parse_fallback_to_bold_header():
    """Falls back to bold header for artist/track when labelled fields are absent."""
    agent = _make_agent()
    recs = agent._parse_recommendations("**Brian Eno — Ambient 1**")
    assert len(recs) == 1
    assert recs[0]["artist"] == "Brian Eno"
    assert recs[0]["track"] == "Ambient 1"


def test_parse_block_with_no_artist_or_track_is_skipped():
    """Blocks that yield neither artist nor track are excluded."""
    agent = _make_agent()
    text = "This is just some intro text with no recommendation blocks."
    recs = agent._parse_recommendations(text)
    assert recs == []


def test_parse_result_preview_is_truncated_in_trace():
    """result_preview in the trace is capped at 500 chars."""
    long_result = "x" * 1000
    tool_resp = _response("tool_use", [_tool_block("search_by_tag", {"tag": "jazz"})])
    final_resp = _response("end_turn", [_text_block(_FINAL_TEXT)])

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[tool_resp, final_resp])

    with (
        patch("src.agent.discovery_agent.anthropic.AsyncAnthropic", return_value=mock_client),
        patch("src.agent.discovery_agent.execute_tool", AsyncMock(return_value=long_result)),
    ):
        agent = MusicDiscoveryAgent()

    import asyncio
    result = asyncio.get_event_loop().run_until_complete(agent.discover("jazz"))
    assert len(result["reasoning_trace"][0]["result_preview"]) <= 500
