import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Page title
st.set_page_config(page_title="News Topic Classifier", layout="centered")
st.title("📰 News Topic Classifier (BERT)")
st.write("Classify a news headline into a topic using a fine-tuned BERT model.")

# Label mapping
labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science / Technology"
}

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("news_topic_bert_model")
    model = BertForSequenceClassification.from_pretrained("news_topic_bert_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# User input
text = st.text_input("Enter a news headline:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        st.success(f"🧠 Predicted Category: **{labels[prediction]}**")


