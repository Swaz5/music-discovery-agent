"""
Music discovery agent.

Orchestrates the Claude API agentic loop: accepts a natural-language user
request, dispatches tool calls to the music service and knowledge base, and
produces a curated list of track recommendations with explanations.

Agentic loop design
-------------------
The loop maintains a growing ``messages`` list that alternates between
``user`` and ``assistant`` roles, exactly as the Anthropic API requires:

    messages = [
        {"role": "user",      "content": user_query},
        {"role": "assistant", "content": [tool_use_block, ...]},  # Claude calls a tool
        {"role": "user",      "content": [tool_result_block, ...]},  # We return the result
        {"role": "assistant", "content": [another_tool_use, ...]},
        {"role": "user",      "content": [another_result, ...]},
        ...
        {"role": "assistant", "content": [final_text_block]},  # stop_reason == end_turn
    ]

The assistant's ``response.content`` (the full list of typed content block
objects) is appended as-is so that tool_use blocks are preserved — the API
requires them to be present in the history when we send back tool_result
blocks on the next turn.
"""

import asyncio
import json
import logging
import re
from typing import Any

import anthropic

from src.agent.tools import TOOLS_REGISTRY, execute_tool, get_tool_schemas
from src.agent.preference_engine import PreferenceEngine

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_ITERATIONS = 15
_MAX_TOKENS = 4096

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
# Identity

You are a music discovery agent with deep knowledge across all genres — from \
mainstream pop to obscure micro-genres. You don't just recommend popular artists; \
you find hidden connections and unexpected sonic matches. You understand that a \
"vibe" is a precise emotional and textural space, and you take that seriously.

# Available tools

| Tool                  | When to use                                                         |
|-----------------------|---------------------------------------------------------------------|
| search_knowledge_base | First call for every request — grounds you in the genre/mood space  |
| search_by_tag         | Find artists within a specific genre tag                            |
| search_artist         | Full profile: bio, tags, similar artists, averaged audio features   |
| get_similar_artists   | Expand beyond obvious picks; often surfaces hidden gems             |
| search_tracks         | Locate a specific track and get its Deezer URL                      |
| get_audio_features    | Numerically verify that a track's energy/valence/tempo fits the vibe|

You don't need to use every tool on every request — be efficient. Aim for \
5–7 tool calls that build on each other rather than 10 redundant ones.

# Discovery strategy

Follow these steps in order:

1. **Ground yourself** — call search_knowledge_base with the user's vibe \
description. Read the results carefully; they contain genre history, typical \
audio characteristics, and canonical artists.

2. **Identify 2–3 tags** — from the knowledge base results, pick the 2–3 \
genre or mood tags that best capture the request (e.g. "trip-hop", \
"dark ambient", "neo-soul").

3. **Find seed artists** — call search_by_tag for each tag. You now have a \
pool of candidates.

4. **Verify audio character** — for the most promising candidates, call \
get_audio_features on a representative track. Check that the numbers match \
the vibe (see Quality rules below). Skip artists whose audio profile \
contradicts the request.

5. **Expand laterally** — call get_similar_artists on 1–2 of the best \
candidates. This is where unexpected, lesser-known recommendations come from.

6. **Grab Deezer links** — call search_tracks for each artist you plan to \
recommend. Use the link from the best-matching track result.

7. **Compile recommendations** — aim for 5–8 total, following the format below.

# Quality rules

- **Diversity cap**: never recommend more than 2 artists from the same genre.
- **Explain the numbers**: if you checked audio features, reference them in \
your "Why" (e.g. "energy 0.72 and valence 0.38 create that tense, forward- \
momentum feel").
- **Mix fame levels**: include at least 2 lesser-known or cult artists \
alongside any well-known picks.
- **Audio veto**: if get_audio_features returns values that contradict the \
request, do not include that track.
  - "Chill" / "relaxed" vibes → energy should be ≤ 0.6
  - "Euphoric" / "energetic" vibes → energy should be ≥ 0.65
  - "Dark" / "melancholic" vibes → valence should be ≤ 0.45
  - "Uplifting" / "happy" vibes → valence should be ≥ 0.55
- **Wild card**: always include at least one recommendation that sits slightly \
outside the obvious genre but shares the emotional texture — and flag it \
explicitly as a wild card.
- **Specific tracks only**: recommend a particular track, not just an artist name.

# Budget awareness

You have at most {budget} tool calls available for this query. Spend them wisely:

- Call `get_audio_features` for **at most 3–4 artists** — only the most promising candidates.
- Call `get_similar_artists` **at most twice** per query.
- **Reserve your last 2–3 iterations** to write your final formatted recommendations.
- If you have already made 10 or more tool calls, **stop exploring immediately** and write your recommendations using what you already know.

Exceeding the budget means your recommendations will be cut off and the user gets nothing. A concise response with 5 great picks beats an exhaustive one that never arrives.

# Response format

After all your tool calls, present your final recommendations. Use this exact \
structure for each entry — the parser depends on it:

**<Artist> — <Track title>**
Artist: <artist name>
Track: <track title>
Genre/tags: <2–4 comma-separated tags>
Why: <1–3 sentences — explain the sonic fit, reference audio features if you \
checked them, and note any wild-card status>
Vibe description: <1–2 evocative sentences painting the listening experience>
Vibe match: <score 0.0–1.0>
Deezer: <URL, or N/A>

Separate each entry with a blank line. Do not include artists you have not \
actually looked up during this session. Quality and honesty over quantity.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class MusicDiscoveryAgent:
    """
    Autonomous music recommendation agent backed by the Anthropic Claude API.

    Uses Claude's tool_use feature to iteratively query Last.fm, Deezer, a
    Librosa audio analyser, and a RAG knowledge base until it has gathered
    enough information to produce personalised recommendations.
    """

    def __init__(
        self,
        preference_engine: PreferenceEngine | None = None,
    ) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._tools = TOOLS_REGISTRY
        self.model = _MODEL
        self.max_iterations = _MAX_ITERATIONS
        self._prefs = preference_engine or PreferenceEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover(self, user_query: str) -> dict:
        """
        Run the agentic discovery loop for a natural-language music request.

        Parameters
        ----------
        user_query : str
            Free-form description of what the user wants to hear, e.g.
            "melancholic indie rock for a rainy afternoon".

        Returns
        -------
        dict with keys:
            recommendations  : list[dict] — parsed artist/track suggestions
            reasoning_trace  : list[dict] — every tool call and result summary
            iterations       : int        — number of API round-trips
            total_tokens     : int        — cumulative token usage
        """
        messages: list[dict] = [{"role": "user", "content": user_query}]
        reasoning_trace: list[dict] = []
        total_tokens = 0
        iterations = 0

        logger.info("Starting discovery for query: %r", user_query)

        while iterations < self.max_iterations:
            iterations += 1
            logger.info("[Iter %d/%d] Calling Claude API …", iterations, self.max_iterations)

            response = await self._client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=self._build_system_prompt(iterations_used=iterations - 1),
                tools=get_tool_schemas(),
                messages=messages,
            )

            usage = response.usage
            turn_tokens = usage.input_tokens + usage.output_tokens
            total_tokens += turn_tokens
            logger.debug(
                "[Iter %d] stop_reason=%s  tokens=+%d (total=%d)",
                iterations, response.stop_reason, turn_tokens, total_tokens,
            )

            # ── Claude is done — extract final answer ──────────────────
            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                logger.info("[Done] Final answer received (%d chars).", len(final_text))
                return {
                    "recommendations": self._parse_recommendations(final_text),
                    "reasoning_trace": reasoning_trace,
                    "iterations": iterations,
                    "total_tokens": total_tokens,
                }

            # ── Claude wants to call tool(s) ───────────────────────────
            if response.stop_reason == "tool_use":
                # Log any reasoning text Claude emitted before the tool call
                for block in response.content:
                    if block.type == "text" and block.text.strip():
                        logger.info("[Reasoning] %s", block.text.strip()[:300])

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                # Append the full assistant content (preserves tool_use blocks
                # which the API requires to be present when we send tool_result)
                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict] = []
                for tool_block in tool_use_blocks:
                    name = tool_block.name
                    args = tool_block.input
                    call_id = tool_block.id

                    args_preview = json.dumps(args, ensure_ascii=False)[:120]
                    logger.info("[Tool] %s(%s)", name, args_preview)

                    try:
                        result = await execute_tool(name, args)
                        result_str = json.dumps(result, ensure_ascii=False, default=str)
                        logger.info("[Result] %s → %s …", name, result_str[:200])

                        reasoning_trace.append({
                            "iteration": iterations,
                            "tool": name,
                            "arguments": args,
                            "result_preview": result_str[:500],
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": result_str,
                        })

                    except Exception as exc:
                        logger.warning("[Tool error] %s: %s: %s", name, type(exc).__name__, exc)
                        reasoning_trace.append({
                            "iteration": iterations,
                            "tool": name,
                            "arguments": args,
                            "error": str(exc),
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": f"Error executing {name}: {exc}",
                            "is_error": True,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            # ── Unexpected stop reason ─────────────────────────────────
            logger.warning("Unexpected stop_reason %r — breaking loop.", response.stop_reason)
            break

        # Max iterations reached — return whatever we have
        logger.warning("Max iterations (%d) reached without end_turn.", self.max_iterations)

        # Try to salvage the last text block from the conversation
        last_text = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = msg["content"]
                if isinstance(content, list):
                    last_text = next(
                        (b.text for b in content if hasattr(b, "type") and b.type == "text"),
                        "",
                    )
                elif isinstance(content, str):
                    last_text = content
                if last_text:
                    break

        return {
            "recommendations": self._parse_recommendations(last_text),
            "reasoning_trace": reasoning_trace,
            "iterations": iterations,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, iterations_used: int = 0) -> str:
        """
        Return the system prompt for this API call.

        The budget figure in the prompt is updated each iteration so Claude
        always sees exactly how many tool calls remain.  If the user has saved
        preferences, the taste profile is appended after the base prompt.
        """
        remaining = max(0, _MAX_ITERATIONS - iterations_used)
        prompt = _SYSTEM_PROMPT.format(budget=remaining)
        taste_context = self._prefs.get_recommendation_context()
        if not taste_context:
            return prompt
        return prompt + "\n\n" + taste_context

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_recommendations(self, text: str) -> list[dict]:
        """
        Parse Claude's final text response into structured recommendation dicts.

        Expects the format emitted by the system prompt:

            **Artist — Track title**
            Artist: ...
            Track: ...
            Genre/tags: ...
            Why: ...  (may span multiple lines)
            Vibe description: ...  (may span multiple lines)
            Vibe match: 0.0–1.0
            Deezer: <URL or N/A>

        Falls back to heuristic header extraction when labels are missing.
        Returns dicts with keys:
            artist, track, genre_tags, why, vibe_description,
            vibe_match_score, deezer_url
        """
        if not text.strip():
            return []

        recommendations: list[dict] = []

        # All labelled field names — used to bound multi-line value extraction
        _FIELD_BOUNDARY = (
            r'(?:Artist|Track|Genre/tags|Genre|Tags|Why|'
            r'Vibe description|Vibe match|Deezer)'
        )

        # Split on bold "Artist — Track" headers (with optional leading number)
        # Supports em-dash (—), en-dash (–), and plain hyphen (-)
        blocks = re.split(
            r'\n(?=\*\*[^*]+\s*[—–\-]\s*[^*]+\*\*|\d+\.\s+\*\*)',
            text.strip(),
        )

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            rec: dict[str, Any] = {
                "artist": "",
                "track": "",
                "genre_tags": [],
                "why": "",
                "vibe_description": "",
                "vibe_match_score": 0.0,
                "deezer_url": "",
            }

            # ── Single-line fields ────────────────────────────────────

            m = re.search(r'^Artist:\s*(.+)$', block, re.MULTILINE)
            if m:
                rec["artist"] = m.group(1).strip()

            m = re.search(r'^Track:\s*(.+)$', block, re.MULTILINE)
            if m:
                rec["track"] = m.group(1).strip()

            m = re.search(r'^(?:Genre/tags|Genre|Tags):\s*(.+)$', block, re.MULTILINE | re.IGNORECASE)
            if m:
                raw_tags = m.group(1).strip()
                rec["genre_tags"] = [t.strip() for t in re.split(r'[,/]', raw_tags) if t.strip()]

            m = re.search(r'Vibe match:\s*([\d.]+)', block, re.IGNORECASE)
            if m:
                try:
                    raw = float(m.group(1))
                    # Normalise: guard against "8" written instead of "0.8"
                    rec["vibe_match_score"] = round(raw / 10 if raw > 1 else raw, 2)
                except ValueError:
                    pass

            m = re.search(r'Deezer:\s*(https?://\S+)', block, re.IGNORECASE)
            if m:
                rec["deezer_url"] = m.group(1).strip()
            else:
                m = re.search(r'https?://(?:www\.)?deezer\.com/\S+', block)
                if m:
                    rec["deezer_url"] = m.group(0)

            # ── Multi-line fields (run until next labelled field or end) ─

            _ml = re.MULTILINE | re.DOTALL
            _bound = rf'(?=\n{_FIELD_BOUNDARY}:|$)'

            m = re.search(rf'^Why:\s*(.+?){_bound}', block, _ml)
            if m:
                rec["why"] = m.group(1).strip()

            m = re.search(rf'^Vibe description:\s*(.+?){_bound}', block, _ml)
            if m:
                rec["vibe_description"] = m.group(1).strip()

            # ── Fallback: extract artist/track from bold header ───────

            if not rec["artist"] or not rec["track"]:
                hm = re.search(
                    r'\*\*\s*(?:\d+\.\s*)?(.+?)\s*[—–\-]\s*(.+?)\s*\*\*', block
                )
                if hm:
                    if not rec["artist"]:
                        rec["artist"] = hm.group(1).strip()
                    if not rec["track"]:
                        rec["track"] = hm.group(2).strip()

            # ── Include only blocks with at least an artist or track ──

            if rec["artist"] or rec["track"]:
                recommendations.append(rec)

        return recommendations


# ──────────────────────────────────────────────────────────────────────────────
# __main__
# ──────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    agent = MusicDiscoveryAgent()
    query = "Find me music that sounds like driving through a city at night"

    print(f"\n{'=' * 64}")
    print(f"  Query: {query}")
    print(f"{'=' * 64}\n")

    result = await agent.discover(query)

    print(f"\n{'=' * 64}")
    print(f"  Completed in {result['iterations']} iteration(s) "
          f"({result['total_tokens']:,} tokens)")
    print(f"{'=' * 64}")

    print("\n  Tool calls:")
    for step in result["reasoning_trace"]:
        args_str = json.dumps(step.get("arguments", {}), ensure_ascii=False)[:80]
        status = "ERROR" if "error" in step else "ok"
        print(f"    [{step['iteration']}] {step['tool']}({args_str}) → {status}")

    print(f"\n  Recommendations ({len(result['recommendations'])} found):")
    for i, rec in enumerate(result["recommendations"], 1):
        score = rec.get("vibe_match_score", 0)
        tags = ", ".join(rec.get("genre_tags", [])) or "—"
        url = rec.get("deezer_url", "")
        print(f"\n  {i}. {rec.get('artist', '?')} — {rec.get('track', '?')}")
        print(f"     Tags:       {tags}")
        print(f"     Vibe match: {score}")
        if rec.get("why"):
            print(f"     Why: {rec['why'][:140]}")
        if rec.get("vibe_description"):
            print(f"     Vibe: {rec['vibe_description'][:120]}")
        if url:
            print(f"     Deezer: {url}")


if __name__ == "__main__":
    asyncio.run(_demo())
