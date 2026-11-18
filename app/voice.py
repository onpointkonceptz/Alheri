"""
app/voice.py
Offline Voice Transcription Module for Alheri.
Operates 100% offline using faster-whisper on CPU with int8 quantization.
"""

import os
import time
import tempfile
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel

# Global singleton to avoid reloading model weights per request
_WHISPER_INSTANCE: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    """Returns the singleton WhisperModel instance configured for CPU int8 execution."""
    global _WHISPER_INSTANCE
    if _WHISPER_INSTANCE is None:
        # Load local base model with int8 quantization for ultra-low memory footprint
        _WHISPER_INSTANCE = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            download_root=None  # uses standard HF cache loaded during setup
        )
    return _WHISPER_INSTANCE


def transcribe_file(file_path: str) -> Dict[str, Any]:
    """
    Transcribes an audio file (WAV, MP3, M4A, OGG) completely offline.
    Returns the concatenated transcription, detected language, and timing metrics.
    """
    t0 = time.time()
    model = get_whisper_model()

    segments, info = model.transcribe(
        file_path,
        beam_size=1,
        vad_filter=True,  # filters out dead silence
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    segment_list = []
    full_text_parts = []
    for s in segments:
        segment_list.append({
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip()
        })
        full_text_parts.append(s.text.strip())

    full_text = " ".join(full_text_parts).strip()
    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "duration_seconds": round(float(info.duration), 2),
        "latency_ms": latency_ms,
        "segments": segment_list,
        "offline": True
    }


def transcribe_audio_bytes(data: bytes, original_filename: str = "recording.wav") -> Dict[str, Any]:
    """
    Accepts raw audio bytes from FastAPI UploadFile, writes to a secure local temp file,
    executes transcription, and safely cleans up temporary storage.
    """
    suffix = os.path.splitext(original_filename)[1] or ".wav"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(data)
        temp_file.flush()
        temp_file.close()

        result = transcribe_file(temp_path)
        result["filename"] = original_filename
        return result
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
