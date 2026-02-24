# Osum Dubber

🎯 **Overview**

This project builds a modular Python pipeline that:
- Extracts a 15-second clip from a source training video
- Transcribes English speech
- Translates it into natural Hindi
- Generates Hindi voice using TTS
- Aligns audio duration
- Lip-syncs the video with high fidelity
- Outputs a polished Hindi-dubbed video

The system is designed to handle full-length videos when run on a paid GPU, while the submission processes only the required 15-second segment.

🏗 **Architecture**

The pipeline is divided into two stages for stability and compatibility:

**🔹 Stage 1 – Modern Python (3.12)**
Runs in Google Colab default runtime.
- **ffmpeg** → clip extraction
- **faster-whisper** → transcription
- **NLLB-200** → translation
- **Bark** → Hindi TTS
- **librosa** → duration alignment

**🔹 Stage 2 – Isolated Python 3.10 Environment**
Runs Wav2Lip for lip synchronization.

*Reason*: Wav2Lip requires legacy `numpy` & `librosa` versions incompatible with Python 3.12. Instead of downgrading global dependencies, I isolated it in a Python 3.10 virtual environment — preventing conflicts and preserving stability. This mirrors production ML architecture (stage isolation).

📁 **Final Project Structure**
```text
/content
│
├── Wav2Lip
├── lipsync_env
├── osum-dubber
│   ├── data
│   │   ├── intermediate
│   │   └── output
│   ├── extract.py
│   ├── transcribe.py
│   ├── translate.py
│   ├── voice_clone.py
│   └── dub_video.py
```

🧪 **Try it out!**

A sample Google Colab link is here to test the pipeline:
[Google Colab - Osum Dubber Test Pipeline](https://colab.research.google.com/drive/1GCSL6M6OhUhJj7zHQ8rqRZ8wZWhcD6Sh?usp=sharing)

🚀 **How To Run (Google Colab – Recommended)**

**✅ Step 1 – Enable GPU**
Runtime → Change Runtime Type → GPU

**✅ Step 2 – Clone Repository**
```bash
!git clone <your-github-link>
%cd osum-dubber
```

**✅ Step 3 – Install Core Dependencies**
```bash
!pip install faster-whisper
!pip install transformers sentencepiece sacremoses
!pip install torch torchvision torchaudio
!pip install librosa soundfile
```

**✅ Step 4 – Run Main Pipeline (Up To TTS)**
```bash
!python dub_video.py --input input.mp4 --start 00:00:15 --duration 15
```
This produces:
- `clip.mp4`
- `hindi_final.wav`

🔥 **Lip Sync Setup (Isolated Environment)**

**✅ Step 5 – Install Python 3.10**
```bash
!apt update -y
!apt install python3.10 python3.10-venv -y
!python3.10 -m venv lipsync_env
!lipsync_env/bin/pip install --upgrade pip
```

**✅ Step 6 – Install Wav2Lip Dependencies**
```bash
!lipsync_env/bin/pip install torch torchvision torchaudio
!lipsync_env/bin/pip install opencv-python librosa==0.8.0 scipy tqdm numpy==1.23.5
```

**✅ Step 7 – Clone Wav2Lip**
```bash
!git clone https://github.com/Rudrabha/Wav2Lip.git
```

**✅ Step 8 – Download Model**
```bash
!wget https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth -O Wav2Lip/wav2lip_gan.pth
```

**✅ Step 9 – Create Output Folder**
```bash
!mkdir -p data/output
```

**✅ Step 10 – Run Lip Sync**
```bash
%cd Wav2Lip

!/content/lipsync_env/bin/python inference.py \
--checkpoint_path wav2lip_gan.pth \
--face /content/osum-dubber/data/intermediate/clip.mp4 \
--audio /content/osum-dubber/data/intermediate/hindi_final.wav \
--outfile /content/osum-dubber/data/output/lip_synced.mp4 \
--resize_factor 2
```

🎥 **Final Output**
`data/output/lip_synced.mp4`
This is the submission-ready video segment.

🧠 **Resourcefulness Decisions**
- Used open-source models only
- Avoided paid APIs
- Isolated legacy dependencies instead of downgrading environment
- Used Colab free GPU
- Batched operations for scalability
- Duration alignment ensures perfect sync

💰 **Estimated Cost Per Minute (If Scaled)**
| Stage | Cost |
| :--- | :--- |
| Whisper Large | ~$0.02/min GPU |
| Translation | negligible |
| TTS (Bark) | ~$0.03/min GPU |
| Lip Sync | ~$0.05/min GPU |

**Estimated total:** ~$0.10 per minute on paid GPU.
With optimized batching + A10 GPU, cost can reduce further.

⚙ **How To Scale To 500 Hours Overnight**
- Move pipeline to cloud GPU cluster
- Parallelize by splitting videos into segments
- Use job queue (Celery / Redis)
- Use batched Whisper inference
- Replace Bark with XTTS for faster TTS
- Use VideoReTalking for higher fidelity
- Store intermediate results in S3
- Auto-scale Kubernetes workers

*500 hours overnight requires:*
- 8× A100 GPUs
- Parallel chunk processing
- Distributed job orchestration

🚧 **Known Limitations**
- Wav2Lip slightly blurs face (can be improved with GFPGAN)
- Bark inference is slower for long videos
- Translation can be improved with context-aware batching
- Emotion transfer not preserved

🏆 **Improvements With More Time**
- Integrate VideoReTalking for better realism
- Add face restoration (CodeFormer)
- Improve prosody alignment
- Add sentence-level duration matching
- Build web UI for upload & processing
