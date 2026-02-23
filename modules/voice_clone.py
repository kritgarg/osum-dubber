import json
from pathlib import Path
from bark import generate_audio, preload_models
import soundfile as sf


class VoiceCloner:
    def __init__(self):
        print("[INFO] Loading Bark models...")
        preload_models()

    def generate(self, hindi_json_path, output_wav_path):
        with open(hindi_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hindi_text = data["hindi_text"]

        print("[INFO] Generating Hindi speech using Bark...")
        audio_array = generate_audio(hindi_text)

        Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav_path, audio_array, 24000)

        print(f"[SUCCESS] Hindi audio saved at {output_wav_path}")