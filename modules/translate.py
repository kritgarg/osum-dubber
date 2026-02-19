import json
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-hi", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading translation model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512)

        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text


def translate_file(input_json_path, output_json_path, translator):
    """
    Reads transcript.json and saves Hindi translation.
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    english_text = data["full_text"]

    print("[INFO] Translating to Hindi...")
    hindi_text = translator.translate(english_text)

    output_data = {
        "english_text": english_text,
        "hindi_text": hindi_text
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Translation saved to {output_json_path}")

    return hindi_text
