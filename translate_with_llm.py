# -*- coding: utf-8 -*-
"""
NLP Coding Round Assessment
Assessment 2: Translation with LLM (Facebook NLLB-200)

This script:
1. Loads the cleaned dataset from Assignment 1
2. Selects 100 English sentences
3. Translates them to Hindi using Facebook's NLLB-200-distilled-600M model
4. Computes BLEU, CHRF, and TER scores against reference translations
5. Saves translations to Excel and scores to a .txt file

Model: facebook/nllb-200-distilled-600M
- A 600M parameter multilingual translation model from Meta
- Supports 200+ languages including Hindi (hin_Deva)
- NOT Google Translate or Helsinki-NLP
"""

import sys
import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- Ensure required packages are installed ---
def install_if_missing(package, pip_name=None):
    """Install a package if it's not already available."""
    try:
        __import__(package)
    except ImportError:
        name = pip_name or package
        print(f"Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name, '-q'])

print("Checking dependencies...")
install_if_missing('torch', 'torch')
install_if_missing('transformers', 'transformers')
install_if_missing('sentencepiece', 'sentencepiece')
install_if_missing('sacrebleu', 'sacrebleu')
install_if_missing('pandas', 'pandas')
install_if_missing('openpyxl', 'openpyxl')
print("All dependencies ready.\n")

# --- Main Processing ---
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu
import time

# Configuration
PROJECT_DIR = r'c:\Users\hp\Desktop\IIT Patna NLP Test'
INPUT_FILE = os.path.join(PROJECT_DIR, 'English_Hindi_Dataset_Cleaned.xlsx')
OUTPUT_EXCEL = os.path.join(PROJECT_DIR, 'Assignment2_Translations.xlsx')
OUTPUT_SCORES = os.path.join(PROJECT_DIR, 'translation_scores.txt')
MODEL_NAME = "facebook/nllb-200-distilled-600M"
NUM_SENTENCES = 100
SOURCE_LANG = "eng_Latn"  # English in NLLB format
TARGET_LANG = "hin_Deva"  # Hindi (Devanagari) in NLLB format

print("=" * 65)
print("NLP Assessment 2: Translation with LLM")
print(f"Model: {MODEL_NAME}")
print("=" * 65)

# =====================================================================
# Step 1: Load the cleaned dataset and select 100 sentences
# =====================================================================
print(f"\n[Step 1] Loading cleaned dataset from Assignment 1...")
df_full = pd.read_excel(INPUT_FILE)
print(f"  Total rows in cleaned dataset: {len(df_full):,}")

# Select 100 sentences (using a fixed random seed for reproducibility)
# We pick from different parts of the dataset for diversity
df_sample = df_full.sample(n=NUM_SENTENCES, random_state=42).reset_index(drop=True)

english_sentences = df_sample['English Sentences'].tolist()
reference_hindi = df_sample['Hindi Sentences'].tolist()

print(f"  Selected {NUM_SENTENCES} sentences for translation")
print(f"  Sample English: \"{english_sentences[0][:80]}...\"")
print(f"  Sample Reference Hindi: \"{reference_hindi[0][:80]}...\"")

# =====================================================================
# Step 2: Load the NLLB-200 model
# =====================================================================
print(f"\n[Step 2] Loading NLLB-200 model...")
print(f"  Model: {MODEL_NAME}")
print(f"  This may take a few minutes on first run (downloading ~600M model)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()
load_time = time.time() - start_load
print(f"  Model loaded in {load_time:.1f} seconds")

# =====================================================================
# Step 3: Translate English sentences to Hindi
# =====================================================================
print(f"\n[Step 3] Translating {NUM_SENTENCES} sentences to Hindi...")

translated_hindi = []
start_translate = time.time()

for i, sentence in enumerate(english_sentences):
    # Tokenize with source language
    tokenizer.src_lang = SOURCE_LANG
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translation
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANG),
            max_new_tokens=256
        )

    # Decode the translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    translated_hindi.append(translation)

    # Progress update every 10 sentences
    if (i + 1) % 10 == 0 or (i + 1) == NUM_SENTENCES:
        elapsed = time.time() - start_translate
        rate = (i + 1) / elapsed
        eta = (NUM_SENTENCES - i - 1) / rate if rate > 0 else 0
        print(f"  Translated {i+1}/{NUM_SENTENCES} sentences "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

total_translate_time = time.time() - start_translate
print(f"  Translation complete! Total time: {total_translate_time:.1f}s "
      f"({total_translate_time/NUM_SENTENCES:.1f}s per sentence)")

# =====================================================================
# Step 4: Compute BLEU, CHRF, and TER scores
# =====================================================================
print(f"\n[Step 4] Computing evaluation metrics...")

# BLEU Score
bleu = sacrebleu.corpus_bleu(translated_hindi, [reference_hindi])
print(f"  BLEU Score:  {bleu.score:.2f}")

# CHRF Score
chrf = sacrebleu.corpus_chrf(translated_hindi, [reference_hindi])
print(f"  CHRF Score:  {chrf.score:.2f}")

# TER Score
ter = sacrebleu.corpus_ter(translated_hindi, [reference_hindi])
print(f"  TER Score:   {ter.score:.2f}")

# =====================================================================
# Step 5: Save scores to .txt file
# =====================================================================
print(f"\n[Step 5] Saving scores to {OUTPUT_SCORES}...")

with open(OUTPUT_SCORES, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("NLP Assessment 2: Translation Evaluation Scores\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Model Used: {MODEL_NAME}\n")
    f.write(f"Number of Sentences: {NUM_SENTENCES}\n")
    f.write(f"Source Language: English ({SOURCE_LANG})\n")
    f.write(f"Target Language: Hindi ({TARGET_LANG})\n\n")
    f.write("-" * 40 + "\n")
    f.write("EVALUATION METRICS\n")
    f.write("-" * 40 + "\n\n")
    f.write(f"BLEU Score:  {bleu.score:.2f}\n")
    f.write(f"  - {bleu}\n\n")
    f.write(f"CHRF Score:  {chrf.score:.2f}\n")
    f.write(f"  - {chrf}\n\n")
    f.write(f"TER Score:   {ter.score:.2f}\n")
    f.write(f"  - {ter}\n\n")
    f.write("-" * 40 + "\n")
    f.write("SCORE INTERPRETATION\n")
    f.write("-" * 40 + "\n\n")
    f.write("BLEU (BiLingual Evaluation Understudy):\n")
    f.write("  Range: 0-100 (higher is better)\n")
    f.write("  Measures n-gram overlap between translation and reference.\n")
    f.write("  > 30 is generally considered good for MT systems.\n\n")
    f.write("CHRF (Character n-gram F-score):\n")
    f.write("  Range: 0-100 (higher is better)\n")
    f.write("  Measures character-level n-gram overlap.\n")
    f.write("  More robust than BLEU for morphologically rich languages like Hindi.\n\n")
    f.write("TER (Translation Edit Rate):\n")
    f.write("  Range: 0-100+ (lower is better)\n")
    f.write("  Measures the number of edits needed to transform translation to reference.\n")
    f.write("  < 50 is generally considered acceptable.\n\n")
    f.write("-" * 40 + "\n")
    f.write("SAMPLE TRANSLATIONS\n")
    f.write("-" * 40 + "\n\n")
    for i in range(min(10, NUM_SENTENCES)):
        f.write(f"Sentence {i+1}:\n")
        f.write(f"  English:    {english_sentences[i]}\n")
        f.write(f"  Reference:  {reference_hindi[i]}\n")
        f.write(f"  Generated:  {translated_hindi[i]}\n\n")

print(f"  Scores saved to: {OUTPUT_SCORES}")

# =====================================================================
# Step 6: Save translations to Excel
# =====================================================================
print(f"\n[Step 6] Saving translations to Excel...")

df_output = pd.DataFrame({
    'English Sentences': english_sentences,
    'Hindi Translation (NLLB-200)': translated_hindi
})

df_output.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
print(f"  Excel saved to: {OUTPUT_EXCEL}")

# =====================================================================
# Final Summary
# =====================================================================
print(f"\n{'=' * 65}")
print("FINAL SUMMARY")
print(f"{'=' * 65}")
print(f"  Model:              {MODEL_NAME}")
print(f"  Sentences:          {NUM_SENTENCES}")
print(f"  BLEU Score:         {bleu.score:.2f}")
print(f"  CHRF Score:         {chrf.score:.2f}")
print(f"  TER Score:          {ter.score:.2f}")
print(f"  Translation Time:   {total_translate_time:.1f}s")
print(f"  Excel Output:       {OUTPUT_EXCEL}")
print(f"  Scores File:        {OUTPUT_SCORES}")
print(f"{'=' * 65}")
print("Done!")
