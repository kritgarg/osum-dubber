import argparse
import sys
from pathlib import Path

from modules.extract import extract_clip, extract_audio
from modules.transcribe import load_model, transcribe_audio
from modules.translate import Translator, translate_file

def main():
    parser = argparse.ArgumentParser(description="Extract, transcribe, and translate a video clip.")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--start", default="00:00:15", help="Start time (e.g., 00:00:15)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--device", default="cpu", help="Device to run models on (cpu, cuda)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input video '{args.input}' not found.")
        sys.exit(1)

    # Paths (relative to script execution, works well in Colab)
    # Using Path to ensure cross-platform compatibility
    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clip_path = str(output_dir / "clip.mp4")
    audio_path = str(output_dir / "clip.wav")
    transcript_path = str(output_dir / "transcript.json")
    hindi_path = str(output_dir / "hindi_text.json")
    
    try:
        # Step 1: Extract
        print(f"[*] Extracting {args.duration}s clip starting from {args.start}...")
        extract_clip(str(input_path), clip_path, args.start, args.duration)
        extract_audio(clip_path, audio_path)
        
        # Step 2: Transcribe
        print("[*] Transcribing audio...")
        model = load_model(device=args.device)
        transcribe_audio(audio_path, transcript_path, model)
        
        # Step 3: Translate
        print("[*] Translating transcript to Hindi...")
        translator = Translator(device=args.device)
        translate_file(transcript_path, hindi_path, translator)
        
        print(f"[SUCCESS] Pipeline completed! Translation saved to: {hindi_path}")
    except RuntimeError as re:
        print(f"[ERROR] System command failed: {re}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
