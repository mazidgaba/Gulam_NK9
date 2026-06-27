# Gulam_NK9 — NLP Coding Round Assessment

## IIT Patna NLP Test — English-Hindi Dataset Processing & Translation

This repository contains the complete solution for the NLP Coding Round Assessment, covering two assignments:

1. **Assignment 1**: English-Hindi Dataset Processing and Analysis
2. **Assignment 2**: Translation with LLM (Facebook NLLB-200)

---

## 📁 Repository Structure

```
Gulam_NK9/
├── README.md                              # This file
├── process_dataset.py                     # Assignment 1 — Dataset processing script
├── translate_with_llm.py                  # Assignment 2 — LLM translation script
├── English_Hindi_Dataset_Cleaned.xlsx     # Assignment 1 — Cleaned dataset output
├── Assignment2_Translations.xlsx          # Assignment 2 — LLM translations output
└── translation_scores.txt                 # Assignment 2 — BLEU, CHRF, TER scores
```

---

## Assignment 1: English-Hindi Dataset Processing and Analysis

### Overview

This assignment processes the **Helsinki-NLP/opus-100** English-Hindi parallel corpus from Hugging Face. The script loads, filters, and analyzes sentence pairs based on word count and character count criteria.

### Dataset Source

- **Hugging Face Repository**: [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- **Configuration**: `en-hi` (English-Hindi)
- **Split Used**: `train` (534,319 sentence pairs)

### Processing Pipeline

| Step | Description | Result |
|------|-------------|--------|
| 1 | Load dataset from Hugging Face | **534,319** sentence pairs |
| 2 | Verify minimum 10,000 rows | ✅ Passed |
| 3 | Filter: Word count [5, 70] in both languages | 534,319 → **288,972** rows |
| 4 | Filter: Word count difference [-10, +10] | 288,972 → **258,790** rows |
| 5 | Compute character counts & differences | All rows processed |
| 6 | Export to Excel (.xlsx) | ✅ Saved |

### Output Columns

| # | Column Name | Description |
|---|-------------|-------------|
| 1 | English Sentences | Original English text |
| 2 | Hindi Sentences | Original Hindi text |
| 3 | Word Count (English) | Number of words in English sentence (5–70) |
| 4 | Word Count (Hindi) | Number of words in Hindi sentence (5–70) |
| 5 | Word Count Difference | English word count − Hindi word count (−10 to +10) |
| 6 | Character Count (English) | Number of characters in English sentence |
| 7 | Character Count (Hindi) | Number of characters in Hindi sentence |
| 8 | Character Count Difference | English char count − Hindi char count |

### How to Run

```bash
# Install Python 3.10+ if not already installed

# Run the dataset processing script
python process_dataset.py
```

**Dependencies** (auto-installed by the script):
- `datasets` — Hugging Face dataset loading
- `pandas` — Data manipulation
- `openpyxl` — Excel file writing

---

## Assignment 2: Translation with LLM

### Overview

This assignment translates 100 English sentences from the cleaned dataset (Assignment 1) into Hindi using a Large Language Model, then evaluates the translations using standard MT metrics.

### Model Used

| Property | Value |
|----------|-------|
| **Model** | `facebook/nllb-200-distilled-600M` |
| **Type** | Sequence-to-Sequence Translation Model |
| **Developer** | Meta AI (Facebook) |
| **Parameters** | ~600 Million |
| **Architecture** | Transformer (encoder-decoder) |
| **Languages Supported** | 200+ languages |
| **Source Language Code** | `eng_Latn` (English, Latin script) |
| **Target Language Code** | `hin_Deva` (Hindi, Devanagari script) |

> **Note**: This model is **NOT** Google Translate or Helsinki-NLP. It is Meta's NLLB (No Language Left Behind) model, specifically designed for high-quality multilingual translation.

### Translation Pipeline

1. **Select 100 sentences** randomly (seed=42) from the cleaned Assignment 1 dataset
2. **Load NLLB-200** model and tokenizer from Hugging Face
3. **Translate** each English sentence to Hindi using the model
4. **Evaluate** translations against reference Hindi sentences using BLEU, CHRF, and TER

### Evaluation Metrics & Scores

| Metric | Score | Range | Interpretation |
|--------|-------|-------|----------------|
| **BLEU** | **18.09** | 0–100 (↑ higher is better) | Measures n-gram precision overlap between generated and reference translations |
| **CHRF** | **41.00** | 0–100 (↑ higher is better) | Character-level n-gram F-score; more robust for morphologically rich languages like Hindi |
| **TER** | **72.22** | 0–100+ (↓ lower is better) | Translation Edit Rate; measures minimum edits needed to match reference |

#### Detailed BLEU Breakdown

```
BLEU = 18.09  |  1-gram: 48.7%  |  2-gram: 23.9%  |  3-gram: 13.3%  |  4-gram: 8.5%
Brevity Penalty = 0.951  |  Length Ratio = 0.952  |  Hyp Length = 1617  |  Ref Length = 1699
```

#### Score Interpretation Guide

- **BLEU Score**:
  - `> 40`: Very high quality translation
  - `30–40`: Good quality, understandable translations
  - `20–30`: Acceptable quality with some errors
  - `< 20`: Low quality, may need significant improvement

- **CHRF Score**:
  - `> 50`: Strong character-level overlap
  - `40–50`: Reasonable quality
  - `< 40`: Significant character-level divergence

- **TER Score**:
  - `< 40`: Excellent — few edits needed
  - `40–60`: Acceptable quality
  - `> 60`: High edit rate, translations diverge significantly from reference

### How to Run

```bash
# Ensure Assignment 1 output exists first
# (English_Hindi_Dataset_Cleaned.xlsx must be in the same directory)

# Run the translation script
python translate_with_llm.py
```

**Dependencies** (auto-installed by the script):
- `torch` — PyTorch for model inference
- `transformers` — Hugging Face Transformers library
- `sentencepiece` — Tokenizer for NLLB model
- `sacrebleu` — BLEU, CHRF, and TER metric computation
- `pandas` — Data manipulation
- `openpyxl` — Excel file writing

### Sample Translations

| # | English (Input) | Reference Hindi | NLLB-200 Translation |
|---|----------------|-----------------|----------------------|
| 1 | It's so quiet...where are the kodama? | मैं कच्चे लोहे का ढलाई खाना सूँघ सकता हूँ... | यह इतना शांत है ... कोडामा कहाँ हैं? |
| 2 | And I said: ask forgiveness of your Lord; verily He is ever Most Forgiving. | और मैंने कहा, अपने रब से क्षमा की प्रार्थना करो। निश्चय ही वह बड़ा क्षमाशील है | और मैंने कहा, "अपने रब से क्षमा माँगो। निश्चय ही वह बड़ा क्षमाशील है।" |
| 3 | Formics might have wiped us out the last time, you know. | फॉर्मिकस् हमें पिछली बार बाहर सफाया हो सकता है, तुम्हें पता है. | फोर्मिक्स हमें पिछले बार खत्म कर सकते हैं, आप जानते हैं. |

> Full list of 10 sample translations with references is available in `translation_scores.txt`.

### Output Files

| File | Description |
|------|-------------|
| `Assignment2_Translations.xlsx` | Excel with English sentences (Column A) and model-generated Hindi translations (Column B) |
| `translation_scores.txt` | Detailed BLEU, CHRF, and TER scores with sample translations |

---

## 🛠️ System Requirements

- **Python**: 3.10 or higher
- **RAM**: 4 GB minimum (8 GB recommended for NLLB model)
- **Disk Space**: ~2 GB (for model weights and dataset cache)
- **Internet**: Required for first run (downloading dataset and model)
- **GPU**: Optional (CPU inference is supported, ~2-5 seconds per sentence)

## 📝 Notes

- All scripts auto-install required dependencies on first run
- The dataset is cached after the first download (~100 MB)
- The NLLB model is cached after the first download (~600 MB)
- Random seed (42) is used for reproducible sentence selection in Assignment 2

---

## 👤 Author

**Gulam**  
NLP Coding Round Assessment — IIT Patna
