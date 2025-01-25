import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Step 2: Load and Preprocess the Dataset
def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully!")
    print(df.head())

    # Check the distribution of sentiments
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())

    # Preprocess text data
    nltk.download('punkt')
    nltk.download('stopwords')

    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, mentions, and special characters
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    # Apply preprocessing to the dataset
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing completed!")
    return df

# Step 3: Encode Sentiment Labels
def encode_labels(df):
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
    print("Labels encoded successfully!")
    return df, label_encoder

# Step 4: Split the Dataset
def split_dataset(df):
    X = df['cleaned_text']
    y = df['sentiment_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split into training and testing sets!")
    return X_train, X_test, y_train, y_test

# Step 5: Create PyTorch Dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Step 6: Train the Model
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
    print("Training completed!")

# Step 7: Evaluate the Model
def evaluate_model(model, test_loader, device, label_encoder):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification report
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

# Step 8: Predict Sentiment on New Text
def predict_sentiment(text, model, tokenizer, label_encoder, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    return label_encoder.inverse_transform(pred)[0]

# Main Function
def main():
    # Load and preprocess data
    filepath = "twitter_sentiment_data.csv"  # Replace with your dataset file
    df = load_and_preprocess_data(filepath)

    # Encode labels
    df, label_encoder = encode_labels(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Load tokenizer and create datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TweetDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = TweetDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load pre-trained BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model = model.to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train the model
    train_model(model, train_loader, optimizer, device, epochs=3)

    # Evaluate the model
    evaluate_model(model, test_loader, device, label_encoder)

    # Save the model
    model.save_pretrained("sentiment_analysis_model")
    tokenizer.save_pretrained("sentiment_analysis_tokenizer")
    print("Model and tokenizer saved!")

    # Test the model on a new tweet
    tweet = "I love this product! It's amazing."
    predicted_sentiment = predict_sentiment(tweet, model, tokenizer, label_encoder, device)
    print(f"Predicted Sentiment for '{tweet}': {predicted_sentiment}")

# Run the script
if __name__ == "__main__":
    main()