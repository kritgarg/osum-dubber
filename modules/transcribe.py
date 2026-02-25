import json
import logging
from faster_whisper import WhisperModel
from pathlib import Path


def load_model(model_size: str = "base", device: str = "cuda") -> WhisperModel:
    """
    Loads faster-whisper model.
    """
    model = WhisperModel(
        model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    return model


def transcribe_audio(audio_path: str, output_path: str = None, model: WhisperModel = None) -> tuple[str, str]:
    """
    Transcribes audio, detects language, and saves transcript as JSON.
    Auto-detects language and falls back to English if confidence is low.
    """
    if model is None:
        model = load_model()

    print(f"[*] Starting transcription for {audio_path}...")
    segments, info = model.transcribe(audio_path)

    detected_language = info.language
    language_probability = info.language_probability

    if language_probability < 0.8:
        logging.warning(f"Low language detection confidence ({language_probability:.2f}). Defaulting to 'en'.")
        detected_language = "en"
    else:
        print(f"[*] Detected language: '{detected_language}' with confidence {language_probability:.2f}")

    transcript = ""
    segment_data = []

    for segment in segments:
        transcript += segment.text + " "
        segment_data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    transcript = transcript.strip()

    if output_path is not None:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        output_data = {
            "full_text": transcript,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "segments": segment_data
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    return transcript, detected_language
