import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from datetime import datetime
import requests
from models.bilstm_model import BiLSTMModel
from models.demand_dataset import DemandDataset
import time

class Trainer:
    def __init__(self, db, device):
        self.db = db
        self.device = device
        self.seq_length = 30
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout_rate = 0.2
        self.batch_size = 32
        self.epochs = 50
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        return np.array(sequences), np.array(targets)

    def prepare_data(self, data):
        # Convert SQL data to DataFrame
        df = pd.DataFrame(data, columns=['itemid', 'date', 'total_sales'])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Extract and normalize sales data
        sales_series = df['total_sales'].values.reshape(-1, 1)
        normalized_sales = self.scaler.fit_transform(sales_series)
        
        # Create sequences
        X_train, y_train = self.create_sequences(normalized_sales, self.seq_length)
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        return X_train, y_train

    def train_model(self, model, train_loader, criterion, optimizer):
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"<!> Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    def exec(self, payload):
        item_id = payload['item_id']
        save_path = payload.get('save_path', f'lstm_model_item_{item_id}.pth')
        
        data = self.db.get_item_data(item_id)

        if not data:
            raise ValueError(f"No data found for item_id {item_id}")

        X_train, y_train = self.prepare_data(data)
        train_dataset = DemandDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        model = BiLSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        print(f"<!> Starting training for item_id {item_id}")
        self.train_model(model, train_loader, criterion, optimizer)

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': self.scaler,
            'item_id': item_id,
            'training_date': datetime.now().isoformat(),
        }, save_path)
        print(f"<!> Model saved to {save_path}")
        time.sleep(600)
        
        host = f"http://{os.getenv('CDN_HOST')}"
        path = f"{os.getenv('CDN_MODELS_PATH')}/lstm_model_item_{item_id}.pth"
        auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
        url = f"{host}/{path}"

        print(f"<!> Uploading model to {host}/{path}")

        with open(save_path, 'rb') as f:
            headers = {"Content-Type": "application/octet-stream"}
            response = requests.put(url, headers=headers, data=f, auth=auth)
            response.raise_for_status()
            f.close()
        
        os.remove(save_path)

        print(f"<!> Model saved to {host}/{path}")

        return payload