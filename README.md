# Real-Time Content Moderation System

A multi-label toxic comment classifier that detects toxic, obscene, threatening, 
and hateful content, built end-to-end from data exploration through a fine-tuned 
transformer model. Built as a learning project to develop practical ML/NLP skills.

## Problem
Online platforms need to detect policy-violating content (toxicity, threats, 
identity-based hate, etc.) before it reaches users. This project builds and 
compares two approaches to multi-label toxic comment classification.

## Dataset
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) 
(159,571 Wikipedia talk-page comments, 6 overlapping labels: toxic, severe_toxic, 
obscene, threat, insult, identity_hate). Not included in this repo due to size — 
download from the Kaggle link above.

## Approach

### 1. Baseline: TF-IDF + Logistic Regression (One-vs-Rest)
- Custom text cleaning pipeline (regex-based URL/IP removal, punctuation handling, 
  stopword removal)
- TF-IDF vectorization (10,000 features)
- One-vs-Rest Logistic Regression per label

### 2. Fine-tuned DistilBERT
- HuggingFace `transformers`, multi-label classification head
- Subword tokenization (max_length=256, chosen based on measured token-length 
  distribution — ~19% of comments exceeded 128 tokens)
- 2 epochs, learning rate 2e-5, best checkpoint selected by validation loss

## Results

| Category | Baseline Recall | DistilBERT Recall |
|---|---|---|
| toxic | 0.62 | 0.78 |
| severe_toxic | 0.28 | 0.27 |
| obscene | 0.63 | 0.86 |
| threat | 0.16 | 0.34 |
| insult | 0.51 | 0.77 |
| identity_hate | 0.16 | 0.62 |

DistilBERT substantially improved recall on rare/high-severity categories 
(threat, identity_hate) — the categories where missing a real violation is most 
costly. `severe_toxic` remained a weak point for both models, likely due to 
label overlap with `toxic` and limited positive examples (~1% of the dataset).

## Key Engineering Decisions
- **max_length=256** chosen over the default 128 based on measured token-length 
  distribution, balancing truncation risk against training compute cost
- **Raw text (not cleaned text) used for DistilBERT** — transformer tokenizers 
  are pretrained on natural text; the custom cleaning pipeline built for the 
  TF-IDF baseline actively hurts transformer performance
- **Recall prioritized over precision for `threat`** — in a moderation system, 
  missing a real threat is more costly than a false alarm

## What's Next
- FastAPI serving layer for real-time inference
- ONNX export + quantization for latency optimization
- Docker containerization and deployment
- Fairness audit across identity subgroups

## Tech Stack
Python, Pandas, NumPy, scikit-learn, PyTorch, HuggingFace Transformers, 
Google Colab
