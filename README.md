# Osum Dubber

A tool to dub videos using AI. The pipeline includes:
1. Video extraction and audio separation using FFmpeg.
2. Transcription using Faster-Whisper.
3. Translation to Hindi using NLLB-200.
4. Text-to-Speech (TTS) - Pending integration.

## Usage

1. Extract video and audio:
   ```bash
   python check_extract.py
   ```
2. Transcribe audio:
   ```bash
   python check_transcribe.py
   ```
3. Translate transcript:
   ```bash
   python check_translate.py
   ```
