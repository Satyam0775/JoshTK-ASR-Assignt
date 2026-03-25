# Hindi ASR Assignment — Whisper Fine-Tuning and Speech Analysis

## Overview

This project is an end-to-end Automatic Speech Recognition (ASR) assignment focused on Hindi conversational speech. It covers four tasks: fine-tuning the Whisper-small model on a Hindi dataset, detecting speech disfluencies, classifying Hindi spelling correctness using rule-based methods, and implementing a lattice-based fair WER evaluation framework.

The work was completed as part of an AI Researcher Intern assignment for the Speech and Audio domain.

---

## Tasks Completed

### 1. Whisper Fine-Tuning for Hindi ASR

- Processed approximately 10 hours of Hindi conversational speech from a cloud-hosted dataset
- Audio files were resampled to 16 kHz mono using librosa
- Transcriptions were downloaded from JSON files, parsed at the segment level, and concatenated into full transcripts per recording
- The processed dataset was converted to HuggingFace Dataset format
- Whisper-small was fine-tuned for 2 epochs with mixed precision on Google Colab (GPU)
- Evaluation was performed on the FLEURS Hindi test split (418 samples) using forced decoder IDs (`language=hi`, `task=transcribe`)
- Both reference and predicted transcripts were normalized before WER computation

### 2. Speech Disfluency Detection

- Each transcription JSON file contained utterance-level segments with start and end timestamps
- A curated list of Hindi disfluency tokens was defined, covering filler sounds (ह्म्म, उम्म, हाँ, जी), repetitions, false starts, and hesitation sounds
- Each segment was scanned using keyword matching; flagged segments were tagged with the detected disfluency type
- Audio clips were extracted from the full recordings using the segment timestamps and saved as individual WAV files
- Results were saved to a structured CSV file

### 3. Hindi Spelling Classification

- Input dataset contained approximately 1,77,000 unique words from a human-transcribed Hindi corpus
- After cleaning (retaining only valid Devanagari Unicode characters and removing duplicates), 1,62,211 words remained
- A multi-rule heuristic classifier was applied in priority order:
  1. Frequency threshold using the `wordfreq` library (Zipf score >= 1.5 classified as correct)
  2. Short-word rule (2 or fewer characters treated as correct)
  3. Filler and noise detection using regex on repetitive consonant patterns
  4. Compound word detection by splitting and checking sub-part frequencies
  5. English words transcribed in Devanagari treated as correctly spelled
  6. Valid Hindi grammatical suffixes treated as correct
  7. Zero-frequency words with 7 or more characters classified as likely misspelled

### 4. Lattice-Based Fair WER Evaluation

- Outputs from six ASR models were aligned at the word level to construct a confusion network
- A consensus reference transcript was generated using majority voting (3 or more out of 6 models agreeing on a word)
- Fair WER for each model was computed as the minimum of the WER against the original human reference and the WER against the consensus reference
- This approach ensures models are not penalized for differences that are attributable to reference errors

---

## Results

### Q1 — Whisper Fine-Tuning

| Model | WER Score |
|---|---|
| Pretrained Whisper-small (baseline) | 0.8291 |
| Fine-tuned Whisper-small | 0.3184 |

WER reduction: approximately 61.6%

### Q3 — Spelling Classification

| Category | Count | Percentage |
|---|---|---|
| Total unique words (after cleaning) | 1,62,211 | 100% |
| Correctly spelled | 1,12,584 | 69.4% |
| Incorrectly spelled | 49,627 | 30.6% |

### Q4 — Lattice-Based Fair WER

| Model | WER (Original Ref) | WER (Consensus Ref) | WER (Fair Final) |
|---|---|---|---|
| MODEL_H | 0.0298 | 0.0426 | 0.0191 |
| MODEL_I | 0.0061 | 0.0531 | 0.0061 |
| MODEL_K | 0.0882 | 0.0952 | 0.0818 |
| MODEL_L | 0.0761 | 0.0947 | 0.0712 |
| MODEL_M | 0.1581 | 0.1721 | 0.1539 |
| MODEL_N | 0.0837 | 0.0937 | 0.0723 |

5 out of 6 models had their WER reduced. MODEL_I achieved the best fair WER of 0.0061.

---

## Tech Stack

- Python
- OpenAI Whisper
- Hugging Face Transformers
- Librosa
- Pandas
- NumPy
- wordfreq
- jiwer
- Google Colab (GPU environment)

---

## Project Structure

```
project/
├── data/
│   ├── downloaded_audio/
│   ├── downloaded_transcripts/
│   └── processed_training_data/
├── Q1_ASR_Finetuning/
│   ├── preprocessing.ipynb
│   ├── prepare_dataset.ipynb
│   ├── whisper_training.ipynb
│   ├── evaluate_fleurs.ipynb
│   └── results/
│       └── wer_results.csv
├── Q2_Disfluency_Detection/
│   ├── detect_disfluency.ipynb
│   ├── clip_audio_segments.ipynb
│   ├── clipped_audio/
│   └── Speech_Disfluencies_Result.csv
├── Q3_Spelling_Correction/
├── Q4_Lattice_Based_WER/
├── models/
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For model training, a GPU environment is recommended. Google Colab with a GPU runtime was used for this project.

---

## How to Run

**Step 1 — Data Preparation**

Run `preprocessing.ipynb` to download audio files from GCP links, resample to 16 kHz mono, and parse JSON transcription files. Then run `prepare_dataset.ipynb` to build the HuggingFace dataset from the processed CSV.

**Step 2 — Fine-Tuning**

Run `whisper_training.ipynb` to fine-tune Whisper-small on the prepared dataset. Training runs for 2 epochs with mixed precision. The fine-tuned model weights are saved to the models directory.

**Step 3 — Evaluation**

Run `evaluate_fleurs.ipynb` to evaluate both the baseline and fine-tuned models on the FLEURS Hindi test split and compute WER scores.

**Step 4 — Disfluency Detection**

Run `detect_disfluency.ipynb` to scan transcription segments for disfluency tokens. Then run `clip_audio_segments.ipynb` to extract and save the corresponding audio clips.

**Step 5 — Spelling Classification and Fair WER**

Run the notebooks under Q3 and Q4 to apply the spelling classification rules and compute lattice-based fair WER scores across models.

---

## Learnings

- Domain-specific fine-tuning on even a modest amount of data (10 hours) significantly improves WER for a target language and domain
- Forced decoder IDs are important when evaluating multilingual models to prevent language switching
- Rule-based NLP can be effective for structured classification tasks when the data characteristics are well understood
- Handling English words transcribed in Devanagari required a dedicated rule to avoid false negatives in spelling classification
- WER is sensitive to reference quality; consensus-based evaluation provides a more reliable measure when multiple model outputs are available

---

## Future Work

- Train on a larger Hindi conversational dataset to improve generalization
- Experiment with larger Whisper variants (medium or large) for potentially lower WER
- Replace keyword-based disfluency detection with a sequence labeling model
- Replace the rule-based spelling classifier with a trained model
- Explore character-level or subword alignment for lattice construction

---

## Author

Satya Kumar Feroma  
AI/ML Enthusiast — Speech and Audio
