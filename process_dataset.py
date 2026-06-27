# -*- coding: utf-8 -*-
"""
NLP Coding Round Assessment
Assessment 1: English-Hindi Dataset Processing and Analysis

This script:
1. Downloads the Helsinki-NLP/opus-100 English-Hindi dataset from Hugging Face
2. Extracts English and Hindi sentence pairs (at least 10,000 rows)
3. Filters by word count (5 to 70 words in both languages)
4. Filters by word count difference (-10 to +10)
5. Computes character counts and differences
6. Exports the cleaned dataset to Excel (.xlsx)
"""

import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import subprocess

# --- Ensure required packages are installed ---
def install_packages():
    required = ['datasets', 'pandas', 'openpyxl']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

install_packages()

# --- Main Processing ---
import pandas as pd
from datasets import load_dataset

print("=" * 60)
print("NLP Assessment 1: English-Hindi Dataset Processing")
print("=" * 60)

# Step 1: Load the dataset from Hugging Face
print("\n[Step 1] Loading Helsinki-NLP/opus-100 (en-hi) dataset...")
dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi")

# Step 2: Extract English and Hindi sentences from the train split
print("[Step 2] Extracting English and Hindi sentence pairs...")
train_data = dataset['train']

english_sentences = []
hindi_sentences = []

for example in train_data:
    translation = example['translation']
    english_sentences.append(translation['en'])
    hindi_sentences.append(translation['hi'])

print(f"  Total sentence pairs extracted: {len(english_sentences):,}")

# Verify we have at least 10,000 rows
if len(english_sentences) < 10000:
    print(f"  WARNING: Only {len(english_sentences)} rows found. Need at least 10,000.")
else:
    print(f"  [OK] Dataset has {len(english_sentences):,} rows (>= 10,000 requirement met)")

# Create a DataFrame
df = pd.DataFrame({
    'English Sentences': english_sentences,
    'Hindi Sentences': hindi_sentences
})

# Step 3: Word Count Analysis
print("\n[Step 3] Computing word counts...")
df['Word Count (English)'] = df['English Sentences'].apply(lambda x: len(str(x).split()))
df['Word Count (Hindi)'] = df['Hindi Sentences'].apply(lambda x: len(str(x).split()))

print(f"  English word count range: {df['Word Count (English)'].min()} - {df['Word Count (English)'].max()}")
print(f"  Hindi word count range: {df['Word Count (Hindi)'].min()} - {df['Word Count (Hindi)'].max()}")

# Filter: Keep only rows where BOTH word counts are between 5 and 70 (inclusive)
before_filter = len(df)
df = df[
    (df['Word Count (English)'] >= 5) & (df['Word Count (English)'] <= 70) &
    (df['Word Count (Hindi)'] >= 5) & (df['Word Count (Hindi)'] <= 70)
]
after_filter = len(df)
print(f"  Filtered by word count [5, 70]: {before_filter:,} → {after_filter:,} rows ({before_filter - after_filter:,} removed)")

# Step 4: Word Count Difference Calculation
print("\n[Step 4] Computing word count differences...")
df['Word Count Difference'] = df['Word Count (English)'] - df['Word Count (Hindi)']

print(f"  Word count difference range: {df['Word Count Difference'].min()} to {df['Word Count Difference'].max()}")

# Filter: Keep only rows where word count difference is between -10 and +10 (inclusive)
before_filter = len(df)
df = df[
    (df['Word Count Difference'] >= -10) & (df['Word Count Difference'] <= 10)
]
after_filter = len(df)
print(f"  Filtered by word count difference [-10, +10]: {before_filter:,} → {after_filter:,} rows ({before_filter - after_filter:,} removed)")

# Step 5: Compute Character Counts
print("\n[Step 5] Computing character counts...")
df['Character Count (English)'] = df['English Sentences'].apply(lambda x: len(str(x)))
df['Character Count (Hindi)'] = df['Hindi Sentences'].apply(lambda x: len(str(x)))
df['Character Count Difference'] = df['Character Count (English)'] - df['Character Count (Hindi)']

# Reset index for clean output
df = df.reset_index(drop=True)

# Reorder columns to match expected output
df = df[[
    'English Sentences',
    'Hindi Sentences',
    'Word Count (English)',
    'Word Count (Hindi)',
    'Word Count Difference',
    'Character Count (English)',
    'Character Count (Hindi)',
    'Character Count Difference'
]]

# Step 6: Export to Excel
print("\n[Step 6] Exporting to Excel...")
output_file = r'c:\Users\hp\Desktop\IIT Patna NLP Test\English_Hindi_Dataset_Cleaned.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"\n{'=' * 60}")
print(f"FINAL RESULTS")
print(f"{'=' * 60}")
print(f"  Total cleaned sentence pairs: {len(df):,}")
print(f"  Output file: {output_file}")
print(f"\n  Column Summary:")
print(f"  {'Column':<45} {'Non-Null':>10}")
print(f"  {'-'*55}")
for col in df.columns:
    print(f"  {col:<45} {df[col].notna().sum():>10,}")

print(f"\n  Sample data (first 5 rows):")
print(df.head().to_string(index=False))
print(f"\n{'=' * 60}")
print("Done! File saved successfully.")
