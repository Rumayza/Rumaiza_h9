import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.data.preprocess_data import preprocess_data 
from src.model.LSTMNet import LSTMNet 
from tqdm import tqdm
import os
from gensim.models import FastText
from sklearn.model_selection import train_test_split

class FastTextDataset(Dataset):
    def __init__(self, X: list, y: np.ndarray, max_seq_length: int = 400):
        self.X = X
        self.y = y
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index]).float()
        y = torch.tensor(self.y[index]).float()
        seq_length = torch.tensor(self.X[index].shape[0])
        
        X = nn.functional.pad(X, (0, 0, 0, self.max_seq_length - X.shape[0]))
        return X, y, seq_length

def convert_to_embeddings(X, ft, max_seq_length):
    emb_dim = ft.vector_size
    return np.array([
        np.array([ft.wv[word] for word in text if word in ft.wv] + [np.zeros(emb_dim)] * (max_seq_length - len(text)))
        for text in X
    ])

def convert_data(df, ft, max_seq_length):
    X = df['sms'].values
    y = df['label'].map({'ham': 0, 'spam': 1}).values.astype(np.float32)  # Map labels to integers
    X = convert_to_embeddings(X, ft, max_seq_length)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    df = preprocess_data('data/dataset.csv')
    
    ft = FastText(vector_size=50, window=3, min_count=1)
    ft.build_vocab(corpus_iterable=df['sms'].values)
    ft.train(corpus_iterable=df['sms'].values, total_examples=len(df['sms']), epochs=5)  # Reduce epochs to speed up training

    max_seq_length = 400
    X_train, X_test, y_train, y_test = convert_data(df, ft, max_seq_length)

    train_dataset = FastTextDataset(X_train, y_train)
    test_dataset = FastTextDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    model = LSTMNet(50, 64, 1, 2, True, 0.2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss().to(device)

    num_epochs = 5  # Reduced for faster training
    history = {'train_loss': [], 'val_loss': []}  # Create a history dictionary to store losses

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch, text_lengths in tqdm(train_dataloader):
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            output_train = model(X_batch, text_lengths).squeeze()
            loss_train = criterion(output_train, y_batch)
            loss_train.backward()
            optimizer.step()
            train_losses.append(loss_train.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)  # Append the average loss for this epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation loss calculation
        model.eval()  # Set the model to evaluation mode
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, text_lengths in test_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output_val = model(X_batch, text_lengths).squeeze()
                loss_val = criterion(output_val, y_batch)
                val_losses.append(loss_val.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)  # Append the average validation loss
        print(f'Validation Loss: {avg_val_loss:.4f}')

        checkpoint_path = f'model/lstm_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)

    # Save the training history to a file
    np.save('training_history.npy', history)  # Save history as a NumPy file

if __name__ == '__main__':
    if not os.path.exists('model'):
        os.makedirs('model')
    train_model()
