# Fact vs Non-Fact Classification

This project is a machine learning application for classifying short text as either **Fact** or **Non-Fact**. It uses a pretrained T5 encoder to convert text into embeddings, then trains an SVM classifier on those embeddings. The final model is exposed through a Streamlit web app.

## Project Overview

The system performs binary sentence-level classification:

- **Fact**: objective or verifiable statements
- **Non-Fact**: subjective, opinion-based, exaggerated, or unsupported statements

The Streamlit app allows users to enter a sentence, run classification, view prediction confidence, inspect SVM decision details, and see recent prediction history.

## Main Features

- Streamlit-based interactive UI
- T5-base encoder for text embeddings
- Attention mean pooling over T5 hidden states
- SVM classifier with balanced class weights
- Hyperparameter tuning using `RandomizedSearchCV`
- Cached training embeddings for faster startup
- Prediction confidence and class probability display
- Recent prediction history inside the app

## Project Files

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application |
| `data1300.csv` | Dataset containing text documents and labels |
| `x_train_t5.npy` | Cached T5 training embeddings |
| `y_train.npy` | Cached training labels |
| `model_results.xlsx` | Model evaluation results |
| `pretrained_embeddings_results.xlsx` | Results from pretrained embedding experiments |
| `finetuned_embeddings_results.xlsx` | Results from finetuned embedding experiments |
| `ml_project_phase8.ipynb` | Project notebook for earlier experiment phase |
| `ml_project_phase9.ipynb` | Project notebook for later experiment phase |
| `ml_project_phase10.ipynb` | Project notebook for later experiment phase |
| `ml_project_phase11.ipynb` | Project notebook for final experiment phase |
| `ML End send report_grp_5.pdf` | Final project report |
| `Fact vs Non-Fact Classification ML Project Team5.pptx` | Project presentation |

## Model Pipeline

1. Input text is tokenized using the T5 tokenizer.
2. The text is passed through the pretrained `t5-base` encoder.
3. Attention mean pooling is applied to produce a fixed-length embedding.
4. The embedding is passed into an SVM classifier.
5. The app returns either **Fact** or **Non-Fact**, along with confidence scores.

## Dataset Format

The dataset file `data1300.csv` contains two columns:

| Column | Description |
| --- | --- |
| `Doc` | Input sentence or text |
| `Label` | Target class: `Fact` or `Non-Fact` |

## Requirements

Install the required Python libraries before running the app:

```bash
pip install streamlit numpy pandas torch transformers scikit-learn scipy sentencepiece