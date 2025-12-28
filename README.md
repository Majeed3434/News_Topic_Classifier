📰 News Topic Classifier Using BERT
📌 Problem Statement & Objective

In today’s digital era, massive volumes of news articles are published daily. Manually categorizing news into topics such as sports, business, or technology is time-consuming and inefficient.

Objective:
The goal of this project is to fine-tune a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model to automatically classify news headlines into predefined topic categories using Natural Language Processing (NLP).

📂 Dataset Loading & Preprocessing
Dataset Used

AG News Dataset

Source: Hugging Face Datasets

Categories:

World

Sports

Business

Science / Technology

Preprocessing Steps

Loaded dataset using datasets.load_dataset

Used BERT tokenizer (bert-base-uncased)

Applied:

Tokenization

Padding to fixed length

Truncation (max length = 128)

Converted data into PyTorch tensors

Used a subset of data to reduce training time on CPU systems

🧠 Model Development & Training
Model

BertForSequenceClassification

Pre-trained model: bert-base-uncased

Number of output labels: 4

Training Details

Transfer learning approach

Optimized for CPU-based training

Key parameters:

Batch size: 4

Epochs: 1

Learning rate: 2e-5

Training handled using Hugging Face Trainer API

📊 Evaluation with Relevant Metrics

The model performance was evaluated using:

Accuracy

Weighted F1-score

These metrics provide a balanced evaluation of classification performance across all categories.

Results (Approximate)

Accuracy: 85–90%

F1-score: ~0.88

🧪 Deployment

The trained model was deployed using Streamlit to enable live interaction.
The trained model is generated locally by running train.py. Due to GitHub file size limitations, the trained weights are not included in the repository.

Features:

User enters a news headline

Model predicts the topic in real time

Simple and interactive web interface

🧾 Final Summary / Insights

Successfully fine-tuned a transformer-based model for news classification

Demonstrated the power of transfer learning using BERT

Achieved high accuracy with limited training data

Built an end-to-end NLP system from training to deployment

Suitable for real-world text classification tasks