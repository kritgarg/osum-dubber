import json
import torch
import re
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        if "nllb" in model_name:
            self.tokenizer.src_lang = "eng_Latn"
            self.tgt_lang = "hin_Deva"

    def split_sentences(self, text):
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s.strip()) > 0]

    def translate_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256,
                num_beams=5,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )

        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def translate(self, text):
        sentences = self.split_sentences(text)
        translated_sentences = []

        for sentence in sentences:
            translated = self.translate_sentence(sentence)
            translated_sentences.append(translated)

        final_text = " ".join(translated_sentences)
        return final_text


def translate_file(input_json_path, output_json_path, translator):

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    english_text = data["full_text"]

    print("[INFO] Translating with improved settings...")
    hindi_text = translator.translate(english_text)

    output_data = {
        "english_text": english_text,
        "hindi_text": hindi_text
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("[SUCCESS] Saved improved translation.")
    return hindi_text
