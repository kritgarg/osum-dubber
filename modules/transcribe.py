import json
from faster_whisper import WhisperModel
from pathlib import Path


def load_model(model_size="base", device="cuda"):
    model = WhisperModel(
        model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    return model


def transcribe_audio(audio_path, output_path, model):
    """
    Transcribes audio and saves transcript as JSON.
    """

    print("[INFO] Transcribing audio...")

    segments, info = model.transcribe(audio_path)

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

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    output_data = {
        "full_text": transcript,
        "segments": segment_data
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Transcript saved to {output_path}")

    return transcript, segment_data
