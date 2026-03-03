# meeting_assistant/output_formatter.py
# ============================================================
# Output formatting:
#   - Parses raw VibeVoice-ASR model output into structured segments
#   - Renders to console (via rich, with plain-text fallback)
#   - Writes JSON files
# ============================================================

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """One transcribed utterance from the model output."""
    speaker: str          # e.g. "Speaker_0", "Speaker_1"
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class TranscriptionResult:
    """Complete result for a single source audio file."""
    source_file: Path
    total_duration_seconds: float
    was_split: bool
    segments: List[Dict]
    transcribed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class OutputFormatter:
    """
    Handles parsing and rendering of VibeVoice-ASR output.

    VibeVoice-ASR generates text with special tokens for speaker identity,
    timestamps, and content. Three parsing strategies are tried in order,
    with plain-text fallback:

      1. Full structured: <|speakerN|><|t=START|> text <|t=END|>
      2. JSON array:      [{"Start":0, "End":5.2, "Speaker":0, "Content":"..."}]
      3. Timestamps only: <|t=START|> text <|t=END|>
      4. Plain text:      everything cleaned of special tokens
    """

    # Pattern 1: Full structured output with speaker + timestamps
    _FULL_PATTERN = re.compile(
        r"<\|speaker(\d+)\|>"        # speaker index
        r"<\|t=([\d.]+)\|>"          # start timestamp
        r"\s*(.*?)\s*"               # utterance text (non-greedy)
        r"<\|t=([\d.]+)\|>",         # end timestamp
        re.DOTALL,
    )

    # Pattern 2: JSON array (some model versions emit this)
    _JSON_ARRAY_PATTERN = re.compile(r"^\s*\[.*\]\s*$", re.DOTALL)

    # Pattern 3: Timestamps only, no speaker
    _TIMESTAMP_ONLY_PATTERN = re.compile(
        r"<\|t=([\d.]+)\|>"
        r"\s*(.*?)\s*"
        r"<\|t=([\d.]+)\|>",
        re.DOTALL,
    )

    def parse_model_output(
        self,
        raw_text: str,
        time_offset: float = 0.0,
    ) -> List[Dict]:
        """
        Parse the raw decoded string from model.generate() into a list of
        segment dicts, applying time_offset for split audio files.
        """
        raw_text = raw_text.strip()

        # Strategy 1: Full structured token format
        matches = self._FULL_PATTERN.findall(raw_text)
        if matches:
            return [
                asdict(Segment(
                    speaker=f"Speaker_{spk}",
                    start_seconds=float(start) + time_offset,
                    end_seconds=float(end) + time_offset,
                    text=text.strip(),
                ))
                for spk, start, text, end in matches
                if text.strip()
            ]

        # Strategy 2: JSON array output
        if self._JSON_ARRAY_PATTERN.match(raw_text):
            try:
                items = json.loads(raw_text)
                if isinstance(items, list) and items:
                    return [
                        asdict(Segment(
                            speaker=f"Speaker_{item.get('Speaker', 0)}",
                            start_seconds=float(item.get("Start", 0.0)) + time_offset,
                            end_seconds=float(item.get("End", 0.0)) + time_offset,
                            text=str(item.get("Content", "")).strip(),
                        ))
                        for item in items
                        if str(item.get("Content", "")).strip()
                    ]
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.debug("JSON parse fallback failed: %s", e)

        # Strategy 3: Timestamps only
        matches = self._TIMESTAMP_ONLY_PATTERN.findall(raw_text)
        if matches:
            return [
                asdict(Segment(
                    speaker="Speaker_0",
                    start_seconds=float(start) + time_offset,
                    end_seconds=float(end) + time_offset,
                    text=text.strip(),
                ))
                for start, text, end in matches
                if text.strip()
            ]

        # Strategy 4: Plain text fallback
        logger.warning(
            "Could not parse structured output from model. "
            "Treating as plain text. Raw preview: %r",
            raw_text[:200],
        )
        cleaned = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()
        if cleaned:
            return [asdict(Segment(
                speaker="Speaker_0",
                start_seconds=time_offset,
                end_seconds=time_offset,
                text=cleaned,
            ))]
        return []

    def write_json(self, result: TranscriptionResult, output_path: Path) -> None:
        """Write the full TranscriptionResult to a JSON file (indent=2)."""
        data = {
            "source_file": str(result.source_file),
            "total_duration_seconds": result.total_duration_seconds,
            "total_duration_human": self._format_duration(result.total_duration_seconds),
            "was_split": result.was_split,
            "transcribed_at": result.transcribed_at,
            "segments": result.segments,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_to_console(self, result: TranscriptionResult) -> None:
        """
        Render the transcription to the terminal.
        Uses rich for color output; falls back to plain print if unavailable.
        """
        try:
            from rich.console import Console
            from rich.text import Text

            console = Console()
            console.rule(f"[bold blue]{result.source_file.name}")
            console.print(
                f"Duration: {self._format_duration(result.total_duration_seconds)}  |  "
                f"Segments: {len(result.segments)}  |  "
                f"Split: {'yes' if result.was_split else 'no'}",
                style="dim",
            )
            console.print()

            speaker_colors = ["cyan", "green", "magenta", "yellow", "blue", "red"]
            for seg in result.segments:
                speaker = seg.get("speaker", "Speaker_0")
                start = seg.get("start_seconds", 0.0)
                end = seg.get("end_seconds", 0.0)
                text = seg.get("text", "")

                try:
                    spk_idx = int(speaker.split("_")[-1])
                except (ValueError, IndexError):
                    spk_idx = 0
                color = speaker_colors[spk_idx % len(speaker_colors)]

                line = Text()
                line.append(f"[{speaker}]", style=f"bold {color}")
                line.append(f"  [{self._ts(start)} → {self._ts(end)}]", style="dim")
                line.append(f"  {text}")
                console.print(line)

            console.print()

        except ImportError:
            print(f"\n=== {result.source_file.name} ===")
            print(
                f"Duration: {self._format_duration(result.total_duration_seconds)}  |  "
                f"Segments: {len(result.segments)}"
            )
            print()
            for seg in result.segments:
                start = seg.get("start_seconds", 0.0)
                end = seg.get("end_seconds", 0.0)
                print(
                    f"[{seg.get('speaker', '?')}] "
                    f"[{self._ts(start)} → {self._ts(end)}] "
                    f"{seg.get('text', '')}"
                )
            print()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Convert seconds to H:MM:SS or M:SS."""
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"

    @staticmethod
    def _ts(seconds: float) -> str:
        """Format seconds as MM:SS.s for console display."""
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:04.1f}"
