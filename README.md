# Anti-Arab Hate Speech Detection System

## NLP Final Project — Spring 2025

### Authors
- Nicholas
- Clara
- Naif 
- Alzmami

---

## Project Overview

This project implements a comprehensive pipeline for detecting hate speech, with a particular emphasis on identifying anti-Arab and anti-Muslim sentiments. It employs text preprocessing techniques, named entity recognition (NER), keyword-based filtering, traditional machine learning models (TF-IDF vectorizer with Logistic Regression), and the state-of-the-art pretrained BERT transformer model from CNERG, (`dehatebert-mono-english`).

The system achieves the following:

- Cleans and preprocesses a labeled dataset of tweets to identify hate speech.
- Implements general hate speech detection using logistic regression and evaluates it with 5-fold cross-validation.
- Develops a specialized anti-Arab hate speech detection model by isolating Arab-related content via keyword matching and NER.
- Conducts detailed performance analyses, including confusion matrices, ROC curves, and error analyses of misclassified examples.
- Compares custom-trained models against a pre-trained BERT model on Arab-specific, non-Arab, and mixed datasets.
- Demonstrates practical deployment through an interactive text-input widget as a lightweight proof-of-concept.

---

## Requirements

This project requires Python 3.8 or greater and should be run entirely in Google Colab without additional configuration.

### Libraries:
- pandas
- numpy
- matplotlib
- nltk
- spacy
- scikit-learn
- transformers
- torch
- ipywidgets

To install any missing dependencies, (if Colab doesn't automatically install them upon the first execution of the import section, run the following:

```bash
!pip install pandas numpy matplotlib nltk spacy scikit-learn transformers torch ipywidgets
!python -m nltk.downloader stopwords
!python -m spacy download en_core_web_sm
```

---

## Usage Instructions

1. Open the provided notebook in [Google Colab](https://colab.research.google.com/drive/1KNUTPU-R82Nawmy3n6-hcmG1LFE7fBGk#scrollTo=17DdBlXKP662).
2. Execute the initial cells or run the full script, until prompted to upload the data file (`data.txt`), then upload it (the file is provided amongst the submitted work).
3. Continue running subsequent notebook cells sequentially (or simply `Run All`). The notebook is fully self-contained after uploading the data.
4. After the training and evaluations complete, an interactive cell at the bottom of the notebook allows users to test the model in real-time by entering any text input to check for Arab-related hate speech using the pretrained BERT model.

**Note:** The dataset file `data.txt` will accompany the notebook submission for successful train / test execution and reproduction of results.

---

## Deliverables

- A general-purpose hate speech detection model evaluated via 5-fold cross-validation.
- A custom-trained anti-Arab hate speech detection model with explicit performance evaluations, thoroughly analyzed
- Comparative performance metrics and visualizations between the custom-trained models and the pretrained BERT model.
- A functional interactive demonstration for practical usage simulation.

---

## Deployment Demonstration

A lightweight Python widget at the end of the notebook provides a practical demonstration of the model’s deployment capability. Users can enter a tweet or sentence, and the system immediately returns results regarding whether the input contains hate speech directed towards Arabs or Muslims. This interactive cell serves as a minimal, functional prototype that illustrates how the developed models could be integrated into real-world moderation systems or content analysis tools.
