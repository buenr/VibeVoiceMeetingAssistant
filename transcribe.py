#!/usr/bin/env python3
# transcribe.py
# ============================================================
# VibeVoice Meeting Transcription App — CLI entrypoint
#
# Runs on Windows with NVIDIA CUDA (RTX 3060 12 GB VRAM).
# Uses scerz/VibeVoice-ASR-4bit (4-bit quantized, ~5-7 GB VRAM).
#
# Usage examples:
#   # Single file — prints transcript to console
#   python transcribe.py --input audio\meeting.mp3
#
#   # Single file — saves JSON to output\
#   python transcribe.py --input audio\meeting.mp3 --output-dir output
#
#   # Batch transcribe a folder
#   python transcribe.py --batch audio\ --output-dir output\
#
#   # Add domain-specific hotwords for better accuracy
#   python transcribe.py --input meeting.mp3 --context "Alice, Bob, Q4 Review"
#
#   # Use flash_attention_2 if you installed flash-attn (saves ~1-2 GB VRAM)
#   python transcribe.py --input meeting.mp3 --attn-impl flash_attention_2
#
#   # Check VRAM before running
#   python transcribe.py --vram-check
# ============================================================

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(debug: bool = False) -> None:
    """Configure rich-powered logging, falling back to standard if rich unavailable."""
    level = logging.DEBUG if debug else logging.INFO
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    except ImportError:
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(message)s",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description=(
            "VibeVoice Meeting Transcription App\n"
            "Transcribes audio using scerz/VibeVoice-ASR-4bit on Windows NVIDIA CUDA.\n"
            "Supports speaker diarization, timestamps, and automatic audio splitting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Single file — console output
  python transcribe.py --input audio\\meeting.mp3

  # Single file — save JSON
  python transcribe.py --input audio\\meeting.mp3 --output-dir output

  # Batch transcribe a folder
  python transcribe.py --batch audio\\ --output-dir output

  # Add domain-specific terms to improve accuracy
  python transcribe.py --input meeting.mp3 --context "Acme Corp, John, Q4"

  # Use flash_attention_2 (if flash-attn is installed, saves ~1-2 GB VRAM)
  python transcribe.py --input meeting.mp3 --attn-impl flash_attention_2

  # Check VRAM (no model loading needed)
  python transcribe.py --vram-check
        """,
    )

    # --- Input options (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        metavar="FILE",
        type=Path,
        help="Path to a single audio file to transcribe.",
    )
    input_group.add_argument(
        "--batch", "-b",
        metavar="FOLDER",
        type=Path,
        help="Path to a folder; all audio files inside are transcribed.",
    )
    input_group.add_argument(
        "--vram-check",
        action="store_true",
        help="Display GPU VRAM info and exit (does not load the model).",
    )

    # --- Output options ---
    parser.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        type=Path,
        default=None,
        help="Directory to save JSON transcription files. "
             "If omitted, output is printed to console only.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save JSON files even when --output-dir is specified.",
    )

    # --- Model options ---
    parser.add_argument(
        "--model-path",
        metavar="MODEL",
        default="scerz/VibeVoice-ASR-4bit",
        help="HuggingFace model ID or local path to the 4-bit model. "
             "Default: scerz/VibeVoice-ASR-4bit",
    )
    parser.add_argument(
        "--language-model",
        metavar="LM",
        default="Qwen/Qwen2.5-7B",
        help="Pretrained LM name for the processor tokenizer. "
             "Default: Qwen/Qwen2.5-7B",
    )
    parser.add_argument(
        "--context",
        metavar="HOTWORDS",
        default=None,
        help="Comma-separated hotwords or context to improve recognition "
             "(e.g., 'Alice Smith, Acme Corp, Q4 Budget').",
    )

    # --- VRAM / performance options ---
    vram_group = parser.add_argument_group("VRAM and performance (RTX 3060 12 GB)")
    vram_group.add_argument(
        "--attn-impl",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention implementation. 'sdpa' (default) works on all Windows setups. "
             "'flash_attention_2' saves ~1-2 GB VRAM but requires flash-attn installed. "
             "'eager' is the slowest fallback.",
    )
    vram_group.add_argument(
        "--split-threshold",
        metavar="MINUTES",
        type=float,
        default=25.0,
        help="Auto-split audio files longer than this many minutes. Default: 25",
    )
    vram_group.add_argument(
        "--segment-duration",
        metavar="MINUTES",
        type=float,
        default=25.0,
        help="Duration of each split segment in minutes. Default: 25",
    )
    vram_group.add_argument(
        "--max-new-tokens",
        metavar="N",
        type=int,
        default=32768,
        help="Maximum tokens to generate per audio segment. Default: 32768",
    )
    vram_group.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for model inference. Default: cuda",
    )

    # --- Misc ---
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )

    return parser


def cmd_vram_check() -> int:
    """Show VRAM info without loading the model."""
    import torch

    if not torch.cuda.is_available():
        print(
            "\n[ERROR] CUDA is not available.\n"
            "Ensure NVIDIA drivers are installed and PyTorch was installed with CUDA support:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
        return 1

    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        console = None

    device_count = torch.cuda.device_count()
    print(f"\nCUDA Devices: {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(i) / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
        free_gb = total_gb - reserved_gb
        status = "OK — sufficient for VibeVoice-ASR-4bit" if free_gb >= 5.0 else "LOW VRAM WARNING"
        print(
            f"\n  GPU {i}: {props.name}\n"
            f"    Total:     {total_gb:.2f} GB\n"
            f"    Reserved:  {reserved_gb:.2f} GB\n"
            f"    Allocated: {allocated_gb:.2f} GB\n"
            f"    Free:      {free_gb:.2f} GB\n"
            f"    Status:    {status}\n"
            f"    Note:      VibeVoice-ASR-4bit uses ~5-7 GB VRAM"
        )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.debug)
    logger = logging.getLogger("vibevoice")

    if args.debug:
        logger.debug("Debug logging enabled.")

    # --- VRAM check mode (no model loading) ---
    if args.vram_check:
        return cmd_vram_check()

    # --- Require an input ---
    if args.input is None and args.batch is None:
        parser.print_help()
        print("\nError: Specify --input FILE or --batch FOLDER.")
        return 1

    # --- Build transcriber ---
    from meeting_assistant.transcriber import MeetingTranscriber, TranscriberConfig
    from meeting_assistant.audio_utils import AudioUtils

    config = TranscriberConfig(
        model_path=args.model_path,
        language_model_pretrained=args.language_model,
        attn_implementation=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        num_beams=1,
        device=args.device,
        context_info=args.context,
    )

    audio_utils = AudioUtils(
        split_threshold_seconds=args.split_threshold * 60,
        segment_duration_seconds=args.segment_duration * 60,
    )

    transcriber = MeetingTranscriber(config=config, audio_utils=audio_utils)
    save_json = (args.output_dir is not None) and (not args.no_json)

    # --- Single file mode ---
    if args.input:
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            return 1

        logger.info("Transcribing: %s", args.input)
        result = transcriber.transcribe_file(
            audio_path=args.input,
            output_dir=args.output_dir,
            save_json=save_json,
        )
        transcriber.formatter.print_to_console(result)
        return 0

    # --- Batch folder mode ---
    if args.batch:
        if not args.batch.is_dir():
            print(f"Error: Not a directory: {args.batch}")
            return 1

        logger.info("Batch transcribing folder: %s", args.batch)
        results = transcriber.transcribe_batch(
            folder=args.batch,
            output_dir=args.output_dir,
            save_json=save_json,
        )

        print(f"\nBatch complete — processed {len(results)} file(s).")
        for result in results:
            transcriber.formatter.print_to_console(result)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
