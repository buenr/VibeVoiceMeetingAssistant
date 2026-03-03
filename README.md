# VibeVoice Meeting Transcription App

A Python CLI for transcribing meeting audio using
[scerz/VibeVoice-ASR-4bit](https://huggingface.co/scerz/VibeVoice-ASR-4bit) —
a 4-bit quantized version of Microsoft's VibeVoice-ASR (9B parameters).

Features speaker diarization, timestamps, automatic audio splitting for long
recordings, and JSON output. Runs directly on Windows with an NVIDIA GPU
(no Docker required).

Licensed under the [Apache License 2.0](LICENSE).

---

## Hardware Requirements

| Component | Minimum               | Recommended           |
|-----------|-----------------------|-----------------------|
| GPU       | NVIDIA 10 GB VRAM     | RTX 3060 12 GB        |
| CUDA      | 12.x                  | 12.4+                 |
| RAM       | 16 GB                 | 32 GB                 |
| Disk      | 20 GB free            | 30 GB (model cache)   |

The 4-bit quantized model uses **~5–7 GB VRAM**, leaving ~5–7 GB headroom on
a 12 GB card for activations.

---

## Setup (Windows)

### Prerequisites

1. **Python 3.10+** — [python.org](https://python.org)
2. **Git** — [git-scm.com](https://git-scm.com)
3. **NVIDIA GPU drivers** with CUDA 12.x support
4. **ffmpeg** — [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Download the Windows build, extract it, and add `ffmpeg\bin\` to your PATH.
   - Required only for audio files longer than 25 minutes.

### One-Shot Setup

```bat
REM Clone the repo
git clone https://github.com/YOUR_USERNAME/VibeVoiceMeetingAssistant.git
cd VibeVoiceMeetingAssistant

REM Run the setup script (creates venv, installs all dependencies)
setup_windows.bat
```

The script will:
- Create a Python virtual environment (`venv\`)
- Install PyTorch with CUDA 12.4 wheels
- Clone and install the VibeVoice package from GitHub
- Install all Python dependencies from `requirements.txt`
- Attempt to install `flash-attn` (optional — falls back to SDPA if unavailable)
- Create `audio\` and `output\` directories

### Manual Setup (if you prefer)

```bat
REM Create and activate venv
python -m venv venv
venv\Scripts\activate

REM Install PyTorch with CUDA 12.4 (do this FIRST)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Install VibeVoice from GitHub
pip install git+https://github.com/microsoft/VibeVoice.git

REM Install remaining dependencies
pip install -r requirements.txt

REM Optional: install flash-attn for lower VRAM usage (may fail on some Windows setups)
pip install flash-attn --no-build-isolation
```

---

## Usage

Activate the virtual environment first:

```bat
venv\Scripts\activate
```

### Check VRAM

```bat
python transcribe.py --vram-check
```

### Transcribe a Single File

```bat
REM Console output only
python transcribe.py --input audio\meeting.mp3

REM Save JSON to output\
python transcribe.py --input audio\meeting.mp3 --output-dir output
```

### Batch Transcribe a Folder

```bat
python transcribe.py --batch audio\ --output-dir output
```

### Add Domain-Specific Hotwords

Improve accuracy on names, product names, and technical terms:

```bat
python transcribe.py --input meeting.mp3 --context "Alice Smith, Acme Corp, Q4 Budget"
```

### Use flash_attention_2 (if installed)

```bat
python transcribe.py --input meeting.mp3 --attn-impl flash_attention_2
```

---

## Command Reference

```
usage: transcribe [-h]
                  [--input FILE | --batch FOLDER | --vram-check]
                  [--output-dir DIR] [--no-json]
                  [--model-path MODEL] [--language-model LM] [--context HOTWORDS]
                  [--attn-impl {sdpa,flash_attention_2,eager}]
                  [--split-threshold MINUTES] [--segment-duration MINUTES]
                  [--max-new-tokens N] [--device DEVICE]
                  [--debug]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | — | Single audio file to transcribe |
| `--batch FOLDER` | — | Transcribe all audio files in a folder |
| `--vram-check` | — | Show GPU VRAM info and exit |
| `--output-dir DIR` | None (console only) | Save JSON files here |
| `--no-json` | False | Skip JSON output |
| `--model-path MODEL` | `scerz/VibeVoice-ASR-4bit` | HuggingFace ID or local path |
| `--language-model LM` | `Qwen/Qwen2.5-7B` | Tokenizer backbone |
| `--context HOTWORDS` | None | Comma-separated hotwords |
| `--attn-impl` | `sdpa` | Attention: `sdpa` (Windows-safe), `flash_attention_2` (faster), `eager` |
| `--split-threshold MINUTES` | 25 | Split files longer than this |
| `--segment-duration MINUTES` | 25 | Duration of each split segment |
| `--max-new-tokens N` | 32768 | Max tokens per segment |
| `--device DEVICE` | `cuda` | PyTorch device |
| `--debug` | False | Verbose logging |

---

## Output Formats

### Console

```
─────────────────── meeting.mp3 ───────────────────
Duration: 45:23  |  Segments: 12  |  Split: yes

[Speaker_0]  [00:00.0 → 00:08.3]  Good morning everyone, let's get started.
[Speaker_1]  [00:09.1 → 00:15.7]  Thanks for joining. Today's agenda covers Q4 results.
[Speaker_0]  [00:16.2 → 00:28.4]  Right, so first let's look at the revenue numbers...
```

### JSON (`output/meeting.json`)

```json
{
  "source_file": "C:\\Users\\you\\audio\\meeting.mp3",
  "total_duration_seconds": 2723.4,
  "total_duration_human": "45:23",
  "was_split": true,
  "transcribed_at": "2026-03-03T09:30:00Z",
  "segments": [
    {
      "speaker": "Speaker_0",
      "start_seconds": 0.0,
      "end_seconds": 8.3,
      "text": "Good morning everyone, let's get started."
    },
    {
      "speaker": "Speaker_1",
      "start_seconds": 9.1,
      "end_seconds": 15.7,
      "text": "Thanks for joining. Today's agenda covers Q4 results."
    }
  ]
}
```

---

## Supported Audio Formats

`.wav` `.mp3` `.m4a` `.mp4` `.flac` `.ogg` `.opus` `.webm` `.aac` `.wma`

Audio is automatically resampled to 16 kHz mono before inference.

---

## VRAM Optimization Tips

- **First run** downloads ~7 GB to `%USERPROFILE%\.cache\huggingface\` — subsequent runs use cache.
- **Default attention** (`--attn-impl sdpa`) is the safest choice on Windows.
  Use `flash_attention_2` only if you successfully installed `flash-attn`.
- **Auto-split** kicks in for files > 25 minutes. Reduce `--split-threshold 15` if you hit OOM.
- **Monitor VRAM** while transcribing: open Task Manager → Performance → GPU.
- **Expected usage**: ~5–7 GB during inference, ~7–9 GB peak on long segments.

---

## Troubleshooting

**`CUDA out of memory`**
- Lower `--split-threshold 15` to use smaller segments
- Do not use `--attn-impl eager` (increases memory)
- Check `python transcribe.py --vram-check` for free VRAM

**`Could not parse structured output from model` (WARNING in logs)**
- The transcript is still saved as plain text — this is a non-fatal warning
- Occurs on very short clips (<5 seconds) or if model output format changes

**`ModuleNotFoundError: No module named 'vibevoice'`**
```bat
pip install git+https://github.com/microsoft/VibeVoice.git
```

**`ffmpeg not found in PATH`**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add `ffmpeg\bin` to your Windows PATH
- Required only for audio files > 25 minutes

**`torch.cuda.is_available()` returns False**
```bat
REM Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Project Structure

```
VibeVoiceMeetingAssistant/
├── transcribe.py                # CLI entrypoint
├── meeting_assistant/
│   ├── __init__.py
│   ├── transcriber.py           # Model loading + inference (lazy init)
│   ├── audio_utils.py           # ffprobe duration, ffmpeg splitting
│   └── output_formatter.py     # Parse model output → JSON / console
├── requirements.txt
├── setup_windows.bat            # One-shot Windows setup
├── audio/                       # Place your audio files here
├── output/                      # JSON transcriptions saved here
└── LICENSE                      # Apache 2.0
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Model licenses:
- [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) — MIT
- [scerz/VibeVoice-ASR-4bit](https://huggingface.co/scerz/VibeVoice-ASR-4bit) — inherits MIT
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) — Apache 2.0
