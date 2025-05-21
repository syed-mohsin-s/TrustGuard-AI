##DATA CLEANINNING 
# Import necessary libraries
import pandas as pd
import re
from imblearn.over_sampling import SMOTE

# Load datasets (example: toxic comments and fake news)
df_toxic = pd.read_csv("jigsaw_toxic_comments.csv")   # e.g., columns: "comment_text", "toxic"
df_fakenews = pd.read_csv("fake_news.csv")              # e.g., columns: "text", "label"

# Standardize column names
df_toxic.rename(columns={"comment_text": "text", "toxic": "label"}, inplace=True)
df_fakenews.rename(columns={"text": "text", "label": "label"}, inplace=True)

# Concatenate datasets
combined_df = pd.concat([df_toxic, df_fakenews], ignore_index=True)

# Define a text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation/special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Clean the text column
combined_df['text'] = combined_df['text'].apply(clean_text)

# Remove duplicates and drop missing values
combined_df.drop_duplicates(subset=["text"], inplace=True)
combined_df.dropna(subset=["text", "label"], inplace=True)

# Handle class imbalance using SMOTE (if labels are numeric; adjust as needed)
# Here we assume that a binary classification (e.g., harmful vs. non-harmful) is used.
X = combined_df['text']
y = combined_df['label']
# For demonstration, converting text to a simple bag-of-words representation.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_vectorized, y)
# For training a deep learning model, you may choose to balance the dataset differently.
# Convert back to a DataFrame if needed.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


##Data Preprocessing & Tokenization

from transformers import AutoTokenizer
from datasets import Dataset

# Use DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert our DataFrame to Hugging Face Dataset (using a subset for demonstration)
dataset = Dataset.from_pandas(combined_df)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Optionally, remove the original "text" column after tokenization
tokenized_dataset = tokenized_dataset.remove_columns(["text"])


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##Model Development & Training

#Custom Model Training with Hugging Face (to later deploy on Vertex AI)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# Determine number of labels; assume binary classification for harmful (1) vs. safe (0)
num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

# Split tokenized_dataset into train and validation sets (80/20 split)
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50
)

# Define metric (accuracy)
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
trainer.evaluate()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# Vertex AI Integration

# Upload your dataset to GCS
gsutil cp combined_cleaned_dataset.csv gs://<YOUR_BUCKET>/data/

# Submit a custom training job using gcloud CLI (customize the following command):
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=moderation-model-training \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/<YOUR_PROJECT>/moderation-training:latest,local-package-path=./,python-module=trainer.task

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Integrating Gemini APIs for Advanced Text Analysis

import requests

def analyze_with_gemini(text):
    GEMINI_API_URL = "https://gemini.googleapis.com/v1/analyze"  # Placeholder URL
    headers = {
        "Authorization": "Bearer YOUR_GEMINI_API_KEY",
        "Content-Type": "application/json"
    }
    payload = {
        "document": {"text": text},
        "features": {"sentiment": {}, "factCheck": {}}
    }
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Example usage:
text_to_analyze = "Some potentially harmful content..."
gemini_result = analyze_with_gemini(text_to_analyze)
print(gemini_result)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Real-Time Deployment with FastAPI & Cloud Run 

               ##FastAPI Inference Endpoint

# Save as main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import requests
from google.cloud import firestore

app = FastAPI()

# Initialize Firestore (ensure you have set GOOGLE_APPLICATION_CREDENTIALS)
db = firestore.Client()

# Load the saved model and tokenizer
model_path = "./final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Gemini API function (as defined earlier)
def analyze_with_gemini(text):
    GEMINI_API_URL = "https://gemini.googleapis.com/v1/analyze"  # Placeholder
    headers = {
        "Authorization": "Bearer YOUR_GEMINI_API_KEY",
        "Content-Type": "application/json"
    }
    payload = {"document": {"text": text}, "features": {"sentiment": {}, "factCheck": {}}}
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

class TextInput(BaseModel):
    content: str

@app.post("/predict")
def predict_text(input: TextInput):
    # Step 1: Get classification result
    classification = classifier(input.content)
    
    # Step 2: For high-risk content, perform additional analysis with Gemini API
    # (Here we assume a simple threshold check; adjust as needed)
    toxicity_score = classification[0].get("score", 0)
    flagged = toxicity_score > 0.7
    
    gemini_analysis = {}
    if flagged:
        gemini_analysis = analyze_with_gemini(input.content)
        # Store flagged content in Firestore for moderator review
        db.collection("flagged_content").add({
            "text": input.content,
            "classification": classification,
            "gemini_analysis": gemini_analysis
        })
    
    return {"flagged": flagged, "classification": classification, "gemini_analysis": gemini_analysis}

# To run locally: uvicorn main:app --reload

    ##Containerization & Deployment to Cloud Run

# Use official Python runtime
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install requirements
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . /app/

# Expose port (default for FastAPI)
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



