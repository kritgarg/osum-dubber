# Osum Dubber

A modular Video Dubbing Pipeline that extracts a video clip, transcribes its audio using Whisper, and translates the transcript to Hindi using NLLB.

## Setup

### Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible in your system `PATH`.

### Installation
1. Clone the repository and navigate to the project root.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Use the `dub_video.py` orchestrator script to run the complete pipeline:

```bash
python dub_video.py --input <path_to_video.mp4> --start 00:00:15 --duration 15
```

### CLI Arguments:
- `--input` (required): Path to the source video file (e.g., `.mp4`).
- `--start`: Start time for the clip extraction (default: `00:00:15`).
- `--duration`: Duration of the clip in seconds (default: `15`).
- `--device`: Compute device (`cpu` or `cuda`, default: `cpu`). Use `cuda` for Google Colab.

## Pipeline Structure
- **Extraction**: Trims the video and extracts a 16kHz mono `.wav` audio file.
- **Transcription**: Uses `faster-whisper` to transcribe the audio.
- **Translation**: Uses `facebook/nllb-200-distilled-600M` to translate text to Hindi.
