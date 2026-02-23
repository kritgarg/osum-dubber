import librosa
import soundfile as sf


def match_duration(reference_audio_path, generated_audio_path, output_path):
    """
    Time-stretches generated audio to match reference duration.
    """

    ref_audio, ref_sr = librosa.load(reference_audio_path, sr=None)
    gen_audio, gen_sr = librosa.load(generated_audio_path, sr=None)

    ref_duration = len(ref_audio) / ref_sr
    gen_duration = len(gen_audio) / gen_sr

    print(f"[INFO] Reference duration: {ref_duration:.2f}s")
    print(f"[INFO] Generated duration: {gen_duration:.2f}s")

    stretch_rate = gen_duration / ref_duration

    print(f"[INFO] Applying stretch rate: {stretch_rate:.4f}")

    aligned_audio = librosa.effects.time_stretch(gen_audio, rate=stretch_rate)

    sf.write(output_path, aligned_audio, gen_sr)

    print(f"[SUCCESS] Duration-aligned audio saved to {output_path}")
