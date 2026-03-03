# meeting_assistant/__init__.py
from meeting_assistant.transcriber import MeetingTranscriber, TranscriberConfig
from meeting_assistant.audio_utils import AudioUtils
from meeting_assistant.output_formatter import OutputFormatter, TranscriptionResult

__all__ = [
    "MeetingTranscriber",
    "TranscriberConfig",
    "AudioUtils",
    "OutputFormatter",
    "TranscriptionResult",
]

__version__ = "0.1.0"
