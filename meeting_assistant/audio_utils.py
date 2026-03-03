# meeting_assistant/audio_utils.py
# ============================================================
# Audio utility functions:
#   - Duration detection via ffprobe
#   - Audio splitting into segments via ffmpeg
#   - Supported format discovery
# ============================================================

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Formats that ffmpeg can decode and librosa can load
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".flac",
    ".ogg", ".opus", ".webm", ".aac", ".wma",
}

# Conservative split threshold — 25 minutes (1500 seconds).
# Keep files at 25 min max to stay safely within the 12 GB VRAM budget
# and the 64K token generation limit on the 4-bit model.
DEFAULT_SPLIT_THRESHOLD_SECONDS: float = 25 * 60   # 1500.0
DEFAULT_SEGMENT_DURATION_SECONDS: float = 25 * 60  # 1500.0


@dataclass
class AudioSegment:
    """Represents a local audio file or a split segment thereof."""
    path: Path
    start_offset_seconds: float = 0.0
    duration_seconds: float = 0.0
    is_temp: bool = False  # True if this file was created by splitting


@dataclass
class AudioFile:
    """Metadata and segment list for a single source audio file."""
    source_path: Path
    duration_seconds: float
    segments: List[AudioSegment] = field(default_factory=list)
    was_split: bool = False


class AudioUtils:
    """
    Wraps ffprobe/ffmpeg for duration detection and audio splitting.

    Audio is split using ffmpeg -c copy (no re-encoding) for speed.
    Temp files are created in the system temp dir and must be cleaned
    up by the caller via cleanup_segments().
    """

    def __init__(
        self,
        split_threshold_seconds: float = DEFAULT_SPLIT_THRESHOLD_SECONDS,
        segment_duration_seconds: float = DEFAULT_SEGMENT_DURATION_SECONDS,
    ) -> None:
        self._split_threshold = split_threshold_seconds
        self._segment_duration = segment_duration_seconds
        self._verify_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, audio_path: Path) -> AudioFile:
        """
        Given a path to an audio file, return an AudioFile with a list
        of AudioSegments ready for transcription.

        Files shorter than the threshold produce one segment pointing at
        the original file (is_temp=False).
        Longer files are split into temp segments (is_temp=True).
        """
        audio_path = audio_path.resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: '{audio_path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        duration = self.get_duration(audio_path)
        logger.info(
            "Audio duration: %.1fs (%.1f min) — %s",
            duration, duration / 60, audio_path.name,
        )

        if duration <= self._split_threshold:
            return AudioFile(
                source_path=audio_path,
                duration_seconds=duration,
                segments=[AudioSegment(
                    path=audio_path,
                    start_offset_seconds=0.0,
                    duration_seconds=duration,
                    is_temp=False,
                )],
                was_split=False,
            )

        logger.info(
            "File exceeds %.0f min threshold — splitting into %.0f-min segments.",
            self._split_threshold / 60,
            self._segment_duration / 60,
        )
        segments = self._split(audio_path, duration)
        return AudioFile(
            source_path=audio_path,
            duration_seconds=duration,
            segments=segments,
            was_split=True,
        )

    def get_duration(self, audio_path: Path) -> float:
        """Return the duration of an audio file in seconds via ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffprobe failed for {audio_path.name}: {e.stderr.strip()}"
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffprobe timed out for {audio_path.name}")

        try:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(
                f"Could not parse ffprobe output for {audio_path.name}: {e}"
            ) from e

    def find_audio_files(self, folder: Path) -> List[Path]:
        """
        Recursively find all supported audio files in a folder.
        Returns a sorted list for deterministic batch ordering.
        """
        found: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            found.extend(folder.rglob(f"*{ext}"))
            found.extend(folder.rglob(f"*{ext.upper()}"))
        return sorted(set(found))

    @staticmethod
    def cleanup_segments(audio_file: AudioFile) -> None:
        """Delete temporary segment files created by splitting."""
        for seg in audio_file.segments:
            if seg.is_temp and seg.path.exists():
                try:
                    seg.path.unlink()
                    logger.debug("Cleaned up temp segment: %s", seg.path.name)
                except OSError as e:
                    logger.warning("Could not delete temp segment %s: %s", seg.path, e)
        # Clean up the temp directory if it's now empty
        if audio_file.segments and audio_file.segments[0].is_temp:
            tmp_dir = audio_file.segments[0].path.parent
            try:
                if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                    tmp_dir.rmdir()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(self, audio_path: Path, total_duration: float) -> List[AudioSegment]:
        """Split audio into fixed-duration segments using ffmpeg -c copy."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="vibevoice_split_"))
        stem = audio_path.stem
        suffix = audio_path.suffix
        output_pattern = str(tmp_dir / f"{stem}_%04d{suffix}")

        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(int(self._segment_duration)),
            "-c", "copy",
            "-reset_timestamps", "1",
            "-avoid_negative_ts", "make_zero",
            "-y",
            output_pattern,
        ]

        logger.debug("ffmpeg split command: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=600,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg split failed for {audio_path.name}: {e.stderr.strip()}"
            ) from e

        segment_files = sorted(tmp_dir.glob(f"{stem}_*{suffix}"))
        if not segment_files:
            raise RuntimeError(f"ffmpeg produced no segments for {audio_path.name}")

        segments: List[AudioSegment] = []
        offset = 0.0
        for seg_path in segment_files:
            try:
                seg_duration = self.get_duration(seg_path)
            except RuntimeError:
                seg_duration = self._segment_duration  # fallback estimate
            segments.append(AudioSegment(
                path=seg_path,
                start_offset_seconds=offset,
                duration_seconds=seg_duration,
                is_temp=True,
            ))
            offset += seg_duration

        logger.info("Split into %d segment(s) in %s", len(segments), tmp_dir)
        return segments

    @staticmethod
    def _verify_ffmpeg() -> None:
        """Fail fast if ffmpeg or ffprobe are not on PATH."""
        missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
        if missing:
            raise EnvironmentError(
                f"{', '.join(missing)} not found in PATH. "
                "Install ffmpeg from https://ffmpeg.org/download.html "
                "and add its bin\\ folder to your PATH."
            )
