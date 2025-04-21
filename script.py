import os
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import DistilBertForSequenceClassification, AdamW


# Loading the datatset

s3_path = 's3://hugging-face-multiclass-textclassification-bucket-custombucket/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t',names=['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP'])

df = df[['TITLE','CATEGORY']]

my_dict = {
    'e':'Entertainment',
    'b':'Business',
    't':'Science',
    'm':'Health'
}

def update_cat(x):
    return my_dict[x]

df['CATEGORY'] = df['CATEGORY'].apply(lambda x:update_cat(x))

# print(df)

# This is just a tip
df = df.sample(frac=0.05,random_state=1)

df = df.reset_index(drop=True)
# Tip ends

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print(f"DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x:encode_cat(x))

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the texts and prepare inputs

# Prepare Dataset for PyTorch
class NewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=20):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the input text and return token ids, attention mask, and label
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['TITLE'].tolist(),
    df['ENCODE_CAT'].tolist(),
    test_size=0.1, random_state=42
)

# Create Dataset objects
train_dataset = NewDataset(train_texts, train_labels, tokenizer)
val_dataset = NewDataset(val_texts, val_labels, tokenizer)

# DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Load pre-trained DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(encode_dict))

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



## Training Loop
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        # Move batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Get predictions
        _, preds = torch.max(logits, dim=1)
        correct_preds += torch.sum(preds == labels)
        total_preds += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds.double() / total_preds

    return avg_loss, accuracy


# Training loop
epochs = 2
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}")
    


def eval_model(model, data_loader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            _, preds = torch.max(logits, dim=1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)
    
    accuracy = correct_preds.double() / total_preds
    return accuracy

val_accuracy = eval_model(model, val_loader, device)
print(f"Validation accuracy: {val_accuracy:.4f}")


# Save model to S3
model.save_pretrained('/opt/ml/model')
tokenizer.save_pretrained('/opt/ml/model')

print("Model saved to the S3 bucket")




