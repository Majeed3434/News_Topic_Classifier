# train.py

from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate
import numpy as np

# 1. Load dataset
print("Loading AG News dataset...")
dataset = load_dataset("ag_news")

# ⬇️ USE SMALL SUBSET (VERY IMPORTANT)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1500))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(300))

# 2. Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. Tokenization function
def tokenize_data(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 4. Apply tokenization
tokenized_dataset = dataset.map(tokenize_data, batched=True)

# 5. Set format
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

train_data = tokenized_dataset["train"]
test_data = tokenized_dataset["test"]

# 6. Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# 7. Load metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

# 8. Training arguments (CPU-OPTIMIZED)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=50,
    save_total_limit=1,
    report_to="none"
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10. Train
print("Training started...")
trainer.train()

# 11. Evaluate
print("Evaluating model...")
results = trainer.evaluate()
print(results)

# 12. Save model
model.save_pretrained("news_topic_bert_model")
tokenizer.save_pretrained("news_topic_bert_model")

print("Model saved successfully!")

