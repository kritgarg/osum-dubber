import argparse
import sys
import json
from pathlib import Path

from modules.extract import extract_clip, extract_audio
from modules.transcribe import load_model, transcribe_audio
from modules.translate import Translator, translate_file
from modules.voice_clone import VoiceCloner
from modules.audio_align import match_duration
from modules.language_map import WHISPER_TO_NLLB

def main():
    parser = argparse.ArgumentParser(description="Extract, transcribe, and translate a video clip to Hindi.")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--start", default="00:00:15", help="Start time (e.g., 00:00:15)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--device", default="cpu", help="Device to run models on (cpu, cuda)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input video '{args.input}' not found.")
        sys.exit(1)

    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clip_path = str(output_dir / "clip.mp4")
    audio_path = str(output_dir / "clip.wav")
    transcript_path = str(output_dir / "transcript.json")
    hindi_path = str(output_dir / "hindi_text.json")
    hindi_wav_path = str(output_dir / "hindi_raw.wav")
    final_output_path = str(output_dir / "hindi_final.wav")
    
    try:
        # Step 1: Extract
        print(f"[*] Extracting {args.duration}s clip starting from {args.start}...")
        extract_clip(str(input_path), clip_path, args.start, args.duration)
        extract_audio(clip_path, audio_path)
        
        # Step 2: Transcribe
        print("[*] Transcribing audio and detecting language...")
        model = load_model(device=args.device)
        full_text, detected_lang = transcribe_audio(audio_path, transcript_path, model)
        
        print(f"[*] Language detected: {detected_lang}")

        # Step 3: Translate
        if detected_lang == "hi":
            print("[*] Detected Hindi. Skipping translation...")
            with open(hindi_path, "w", encoding="utf-8") as f:
                json.dump({
                    "original_text": full_text,
                    "hindi_text": full_text
                }, f, indent=4, ensure_ascii=False)
        else:
            if detected_lang not in WHISPER_TO_NLLB:
                raise ValueError(f"Language '{detected_lang}' is not supported for translation. Please update language_map.py.")
            
            nllb_source_lang = WHISPER_TO_NLLB[detected_lang]
            print(f"[*] Translating transcript from '{nllb_source_lang}' to Hindi...")
            translator = Translator(source_lang=nllb_source_lang, device=args.device)
            translate_file(transcript_path, hindi_path, translator)
            
        # Step 4: Generate Hindi Audio
        print("[*] Generating Hindi audio...")
        cloner = VoiceCloner()
        cloner.generate(
            hindi_json_path=hindi_path,
            output_wav_path=hindi_wav_path
        )
        
        # Step 5: Duration alignment
        print("[*] Aligning audio duration...")
        match_duration(
            reference_audio_path=audio_path,
            generated_audio_path=hindi_wav_path,
            output_path=final_output_path
        )
        
        print(f"[SUCCESS] Pipeline completed! Translation saved to: {hindi_path}")
        print(f"[SUCCESS] Hindi duration-aligned audio saved to: {final_output_path}")
    except RuntimeError as re:
        print(f"[ERROR] System command failed: {re}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
