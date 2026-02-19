import subprocess
from pathlib import Path


def run_command(cmd):
    """
    Runs a shell command safely.
    """
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}") from e


def ensure_directory(path):
    """
    Ensures parent directory exists.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def extract_clip(input_path, output_path, start_time="00:00:15", duration=15):
    """
    Extracts a precise video clip from input video.
    Re-encodes to ensure exact frame accuracy.
    """
    ensure_directory(output_path)

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", input_path,
        "-ss", start_time,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        output_path
    ]

    print(f"[INFO] Extracting clip: {start_time} → {duration}s")
    run_command(cmd)
    print(f"[SUCCESS] Clip saved at: {output_path}")


def extract_audio(video_path, audio_path):
    """
    Extracts mono 16kHz WAV audio for Whisper.
    """
    ensure_directory(audio_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    print("[INFO] Extracting audio (16kHz mono WAV)...")
    run_command(cmd)
    print(f"[SUCCESS] Audio saved at: {audio_path}")


def get_duration(file_path):
    """
    Returns duration (in seconds) using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not get duration for {file_path}")
