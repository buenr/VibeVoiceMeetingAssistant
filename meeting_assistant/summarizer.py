# meeting_assistant/summarizer.py
# ============================================================
# Meeting transcript summarization via Gemini 2.5 Flash Lite.
#
# Uses the google-genai SDK (pip install google-genai).
# API key is read from the GOOGLE_API_KEY environment variable
# or passed explicitly.
# ============================================================

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash-lite"

# ------------------------------------------------------------------ #
# System prompt — instructs Gemini how to structure the summary       #
# ------------------------------------------------------------------ #
_SYSTEM_PROMPT = """\
You are an expert meeting analyst. You receive a meeting transcript that \
includes speaker labels (Speaker_0, Speaker_1, …) and timestamps.

Produce a concise, structured meeting summary. Respond with ONLY valid JSON \
(no markdown fences, no commentary) matching this schema exactly:

{
  "summary":      "string — 2-3 sentence executive overview",
  "duration":     "string — human-readable total duration, e.g. '42 min'",
  "participants": ["Speaker_0", "Speaker_1", ...],
  "topics": [
    {"title": "string", "description": "string — 1-2 sentences"}
  ],
  "action_items": [
    {"owner": "Speaker_N or name if mentioned", "action": "string"}
  ],
  "decisions": ["string"],
  "key_quotes":  [
    {"speaker": "Speaker_N", "timestamp": "MM:SS", "quote": "string"}
  ]
}

Rules:
- Be factual. Do not invent details not present in the transcript.
- If a field has no content (e.g. no decisions), use an empty list [].
- Keep each topic description to 1-2 sentences.
- Limit key_quotes to the 3 most impactful quotes.
- Use the speaker label as-is (Speaker_0, Speaker_1…) unless the transcript \
  reveals their name, in which case use "Name (Speaker_N)".
"""

# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #

@dataclass
class MeetingSummary:
    """Structured output from the Gemini summarization call."""
    summary: str
    duration: str
    participants: List[str]
    topics: List[Dict]
    action_items: List[Dict]
    decisions: List[str]
    key_quotes: List[Dict]
    model: str = GEMINI_MODEL
    raw_json: str = ""

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "summary": self.summary,
            "duration": self.duration,
            "participants": self.participants,
            "topics": self.topics,
            "action_items": self.action_items,
            "decisions": self.decisions,
            "key_quotes": self.key_quotes,
        }

    def to_markdown(self) -> str:
        """Render the summary as human-readable Markdown."""
        lines: List[str] = []

        lines.append(f"## Meeting Summary\n")
        lines.append(f"{self.summary}\n")
        lines.append(f"**Duration:** {self.duration}  ")
        lines.append(f"**Participants:** {', '.join(self.participants)}\n")

        if self.topics:
            lines.append("## Topics Discussed\n")
            for t in self.topics:
                lines.append(f"**{t.get('title', '')}**")
                lines.append(f"{t.get('description', '')}\n")

        if self.action_items:
            lines.append("## Action Items\n")
            for a in self.action_items:
                owner = a.get("owner", "?")
                action = a.get("action", "")
                lines.append(f"- [ ] **{owner}**: {action}")
            lines.append("")

        if self.decisions:
            lines.append("## Decisions Made\n")
            for d in self.decisions:
                lines.append(f"- {d}")
            lines.append("")

        if self.key_quotes:
            lines.append("## Key Quotes\n")
            for q in self.key_quotes:
                speaker = q.get("speaker", "?")
                ts = q.get("timestamp", "")
                quote = q.get("quote", "")
                lines.append(f"> \"{quote}\"")
                lines.append(f"> — {speaker}  [{ts}]\n")

        lines.append(f"---\n*Summarized by {self.model}*")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Summarizer                                                           #
# ------------------------------------------------------------------ #

class MeetingSummarizer:
    """
    Calls Gemini 2.5 Flash Lite to summarize a meeting transcript.

    API key resolution order:
      1. api_key argument passed to __init__
      2. GOOGLE_API_KEY environment variable
      3. GEMINI_API_KEY environment variable (fallback alias)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Gemini API key not found. Set the GOOGLE_API_KEY environment variable "
                "or pass --gemini-api-key KEY to the CLI.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        self._client = None  # lazy init

    def summarize(
        self,
        segments: List[Dict],
        total_duration_seconds: float,
        source_file: str = "",
    ) -> MeetingSummary:
        """
        Format the transcript segments into a prompt and call Gemini.

        Parameters
        ----------
        segments : List[Dict]
            The parsed segments from TranscriptionResult (speaker, start, end, text).
        total_duration_seconds : float
            Total audio duration — embedded in the prompt for context.
        source_file : str
            Name of the source audio file — for context only.

        Returns
        -------
        MeetingSummary
        """
        if not segments:
            raise ValueError("Cannot summarize an empty transcript.")

        prompt = self._build_prompt(segments, total_duration_seconds, source_file)
        raw_response = self._call_gemini(prompt)
        return self._parse_response(raw_response)

    def save(
        self,
        summary: MeetingSummary,
        output_dir: Path,
        stem: str,
        fmt: str = "markdown",
    ) -> Path:
        """
        Save the summary to disk.

        fmt: "markdown" → <stem>_summary.md
             "json"     → <stem>_summary.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            out_path = output_dir / f"{stem}_summary.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            out_path = output_dir / f"{stem}_summary.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary.to_markdown())

        logger.info("Saved summary: %s", out_path)
        return out_path

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _build_prompt(
        self,
        segments: List[Dict],
        total_duration_seconds: float,
        source_file: str,
    ) -> str:
        """
        Format the segments list into a readable transcript block for the prompt.
        """
        minutes = int(total_duration_seconds // 60)
        seconds = int(total_duration_seconds % 60)
        duration_str = f"{minutes}:{seconds:02d}"

        lines = [
            f"Meeting audio: {source_file or 'unknown'}",
            f"Total duration: {duration_str}",
            "",
            "=== TRANSCRIPT ===",
        ]

        for seg in segments:
            speaker = seg.get("speaker", "Speaker_?")
            start = seg.get("start_seconds", 0.0)
            text = seg.get("text", "").strip()
            if not text:
                continue
            # Format timestamp as MM:SS
            mm = int(start // 60)
            ss = int(start % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {speaker}: {text}")

        lines.append("=== END TRANSCRIPT ===")
        return "\n".join(lines)

    def _get_client(self):
        """Lazily initialize the google-genai client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "The 'google-genai' package is required for summarization.\n"
                    "Install it with: pip install google-genai"
                ) from e
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _call_gemini(self, prompt: str) -> str:
        """Send the prompt to Gemini 2.5 Flash Lite and return the raw text."""
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' package is required for summarization.\n"
                "Install it with: pip install google-genai"
            ) from e

        client = self._get_client()
        logger.info("Calling %s for summarization...", GEMINI_MODEL)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,   # low temperature → more deterministic / factual
                max_output_tokens=4096,
            ),
        )

        raw = response.text
        logger.debug("Gemini raw response (%d chars): %r", len(raw), raw[:300])
        return raw

    @staticmethod
    def _parse_response(raw: str) -> MeetingSummary:
        """
        Parse the JSON response from Gemini into a MeetingSummary.
        Strips markdown code fences if the model emits them despite instructions.
        """
        # Strip markdown code fences (```json ... ```) if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(
                "Gemini returned non-JSON output: %s\nRaw: %r", e, raw[:500]
            )
            # Graceful degradation: return a minimal summary with the raw text
            return MeetingSummary(
                summary=raw[:500],
                duration="unknown",
                participants=[],
                topics=[],
                action_items=[],
                decisions=[],
                key_quotes=[],
                raw_json=raw,
            )

        return MeetingSummary(
            summary=data.get("summary", ""),
            duration=data.get("duration", "unknown"),
            participants=data.get("participants", []),
            topics=data.get("topics", []),
            action_items=data.get("action_items", []),
            decisions=data.get("decisions", []),
            key_quotes=data.get("key_quotes", []),
            raw_json=raw,
        )
