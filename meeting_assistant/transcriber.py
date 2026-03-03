# meeting_assistant/transcriber.py
# ============================================================
# Core transcription logic:
#   - Model and processor loading (lazy initialization)
#   - Single-file and batch transcription
#   - 12 GB VRAM optimizations for RTX 3060
# ============================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from meeting_assistant.audio_utils import AudioFile, AudioSegment, AudioUtils
from meeting_assistant.output_formatter import OutputFormatter, TranscriptionResult

logger = logging.getLogger(__name__)

# Default model — 4-bit quantized, pre-built NF4 weights (~5-7 GB VRAM)
DEFAULT_MODEL_PATH = "scerz/VibeVoice-ASR-4bit"
# Tokenizer/processor backbone for the Qwen2.5-7B LLM decoder
DEFAULT_LANGUAGE_MODEL = "Qwen/Qwen2.5-7B"


@dataclass
class TranscriberConfig:
    """
    All user-tunable knobs for the transcription run.

    Defaults are optimized for NVIDIA RTX 3060 12 GB VRAM on Windows:
    - attn_implementation="sdpa": safest on Windows (no flash-attn compile needed).
      Change to "flash_attention_2" if you installed flash-attn for ~1-2 GB less VRAM.
    - temperature=0.0, do_sample=False, num_beams=1: greedy decode (fastest + deterministic).
    - max_new_tokens=32768: matches the model's documented generation ceiling.
    """
    model_path: str = DEFAULT_MODEL_PATH
    language_model_pretrained: str = DEFAULT_LANGUAGE_MODEL
    # Attention backend: "sdpa" (default, Windows-safe) or "flash_attention_2"
    attn_implementation: str = "sdpa"
    max_new_tokens: int = 32768
    temperature: float = 0.0
    do_sample: bool = False
    num_beams: int = 1
    device: str = "cuda"
    # Optional domain-specific hotwords / context text to improve recognition
    context_info: Optional[str] = None


class MeetingTranscriber:
    """
    Manages the VibeVoice-ASR model lifecycle and runs transcription.

    The model is loaded lazily on first use, so startup is fast for --help
    and --vram-check commands. The 4-bit model occupies ~5-7 GB VRAM; only
    one instance should run at a time on a 12 GB card.
    """

    def __init__(
        self,
        config: Optional[TranscriberConfig] = None,
        audio_utils: Optional[AudioUtils] = None,
        output_formatter: Optional[OutputFormatter] = None,
    ) -> None:
        self.config = config or TranscriberConfig()
        self.audio_utils = audio_utils or AudioUtils()
        self.formatter = output_formatter or OutputFormatter()

        self._processor = None
        self._model = None
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_file(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        save_json: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file.

        Files longer than the split threshold are automatically split into
        segments, each transcribed sequentially, then merged.
        Returns a TranscriptionResult and optionally saves JSON.
        """
        self._ensure_model_loaded()

        audio_file = self.audio_utils.prepare(audio_path)
        try:
            result = self._transcribe_audio_file(audio_file)
        finally:
            self.audio_utils.cleanup_segments(audio_file)

        if output_dir is not None and save_json:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / f"{audio_path.stem}.json"
            self.formatter.write_json(result, json_path)
            logger.info("Saved JSON: %s", json_path)

        return result

    def transcribe_batch(
        self,
        folder: Path,
        output_dir: Optional[Path] = None,
        save_json: bool = True,
    ) -> List[TranscriptionResult]:
        """
        Transcribe all supported audio files found recursively in folder.
        Files are processed one at a time (batch_size=1) to stay within
        the 12 GB VRAM budget.
        """
        self._ensure_model_loaded()

        audio_files = self.audio_utils.find_audio_files(folder)
        if not audio_files:
            logger.warning("No supported audio files found in: %s", folder)
            return []

        logger.info("Found %d audio file(s) in %s", len(audio_files), folder)

        results: List[TranscriptionResult] = []
        for i, audio_path in enumerate(audio_files, start=1):
            logger.info("[%d/%d] Transcribing: %s", i, len(audio_files), audio_path.name)
            try:
                result = self.transcribe_file(
                    audio_path,
                    output_dir=output_dir,
                    save_json=save_json,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to transcribe %s: %s", audio_path.name, e)
                continue
            finally:
                # Free intermediate GPU tensors between files
                torch.cuda.empty_cache()

        return results

    def check_vram(self) -> Dict:
        """Return VRAM statistics in GB. Does not load the model."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available — check NVIDIA drivers"}

        device_count = torch.cuda.device_count()
        devices = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            free = total - reserved
            devices.append({
                "index": i,
                "name": props.name,
                "total_gb": round(total, 2),
                "reserved_gb": round(reserved, 2),
                "allocated_gb": round(allocated, 2),
                "free_gb": round(free, 2),
                "sufficient_for_vibevoice": free >= 5.0,
            })
        return {"cuda_device_count": device_count, "devices": devices}

    def unload_model(self) -> None:
        """Free GPU memory by deleting model references and clearing VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._model_loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded and VRAM cleared.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load model and processor on first use (lazy initialization)."""
        if self._model_loaded:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. Ensure NVIDIA drivers and CUDA Toolkit are installed, "
                "and that you installed PyTorch with CUDA support:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )

        logger.info("Loading model: %s", self.config.model_path)
        logger.info("  Backbone LM: %s", self.config.language_model_pretrained)
        logger.info("  Attention:   %s", self.config.attn_implementation)
        logger.info("  Device:      %s", self.config.device)

        t0 = time.perf_counter()

        # Lazy import: avoids triggering vibevoice imports for --help / --vram-check
        try:
            from vibevoice import (
                VibeVoiceASRProcessor,
                VibeVoiceASRForConditionalGeneration,
            )
        except ImportError:
            # Fallback: some versions use a sub-module path
            try:
                from vibevoice.asr import (
                    VibeVoiceASRProcessor,
                    VibeVoiceASRForConditionalGeneration,
                )
            except ImportError as e:
                raise ImportError(
                    "The 'vibevoice' package is required but not installed.\n"
                    "Install it with:\n"
                    "  pip install git+https://github.com/microsoft/VibeVoice.git"
                ) from e

        # Processor: handles 16kHz resampling, feature extraction, tokenization.
        # language_model_pretrained_name points to the Qwen2.5-7B tokenizer.
        self._processor = VibeVoiceASRProcessor.from_pretrained(
            self.config.model_path,
            language_model_pretrained_name=self.config.language_model_pretrained,
            trust_remote_code=True,
        )

        # Model: the weights are already NF4-quantized in the checkpoint.
        # Do NOT pass BitsAndBytesConfig — that would attempt to re-quantize.
        # dtype=bfloat16 sets the compute dtype for the dequantized matmuls.
        # bitsandbytes provides the CUDA dequantization kernels at inference time.
        self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            attn_implementation=self.config.attn_implementation,
            trust_remote_code=True,
        )
        self._model.eval()

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1fs", elapsed)

        # Log VRAM after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info("VRAM allocated after load: %.2f GB", allocated)

        self._model_loaded = True

    def _transcribe_audio_file(self, audio_file: AudioFile) -> TranscriptionResult:
        """Transcribe all segments of an AudioFile and merge results."""
        all_segments: List[Dict] = []

        for idx, segment in enumerate(audio_file.segments):
            logger.info(
                "  Segment %d/%d: %s (offset=%.0fs)",
                idx + 1, len(audio_file.segments),
                segment.path.name,
                segment.start_offset_seconds,
            )
            raw_output = self._run_inference(segment)
            parsed = self.formatter.parse_model_output(
                raw_output,
                time_offset=segment.start_offset_seconds,
            )
            all_segments.extend(parsed)

        return TranscriptionResult(
            source_file=audio_file.source_path,
            total_duration_seconds=audio_file.duration_seconds,
            was_split=audio_file.was_split,
            segments=all_segments,
        )

    def _run_inference(self, segment: AudioSegment) -> str:
        """
        Load audio via librosa, build processor inputs, run model.generate(),
        and return the raw decoded string for the output_formatter to parse.
        """
        import librosa

        # Load and resample to 16kHz mono (required by VibeVoice-ASR processor)
        audio_array, sr = librosa.load(str(segment.path), sr=16000, mono=True)

        # Build processor inputs. context injects hotwords / domain info.
        inputs = self._processor(
            audio=audio_array,
            sampling_rate=sr,
            context=self.config.context_info,
            return_tensors="pt",
        )

        # Move input tensors to the model's device
        device = next(self._model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
            )

        # Decode with skip_special_tokens=False so the output_formatter can
        # extract speaker labels and timestamps from the special tokens.
        return self._processor.decode(output_ids[0], skip_special_tokens=False)
