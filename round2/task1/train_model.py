# train_model.py

import pandas as pd
import torch
import numpy as np
# --- CHANGE 1: Import the Dataset class ---
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the Dataset using Pandas
print("Loading AG News dataset from CSV...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Combine Title and Description into a single 'text' column
train_df['text'] = train_df['Title'] + " " + train_df['Description']
test_df['text'] = test_df['Title'] + " " + test_df['Description']

# The labels are 1-based in the CSV (1, 2, 3, 4), but models expect 0-based (0, 1, 2, 3)
# This works correctly
train_df['label'] = train_df['Class Index'].astype(int) - 1
test_df['label'] = test_df['Class Index'].astype(int) - 1


# --- CHANGE 2: Convert Pandas DataFrame to Hugging Face Dataset ---
print("Converting Pandas DataFrame to Hugging Face Dataset...")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# --- CHANGE 3 (Recommended): Shuffle and select a subset for faster training ---
train_dataset = train_dataset.shuffle(seed=42) # Using 10k samples
test_dataset = test_dataset.shuffle(seed=42)   # Using 1k samples


# 2. Load Tokenizer and Preprocess Data
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # The tokenizer will now process the 'text' column we created
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenizing dataset...")
# This .map() call will now work correctly on the Dataset object
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# 3. Load the Pre-trained Model
num_labels = 4 # World, Sports, Business, Sci/Tech
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 4. Define Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# 5. Set Up the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
 )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)

# 6. Train the Model
print("Starting training...")
trainer.train()

# 7. Evaluate the Model
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 8. Save the Model for Deployment
output_model_dir = "./ag_news_bert_classifier"
print(f"Saving model to {output_model_dir}...")
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

print("Training complete and model saved!")