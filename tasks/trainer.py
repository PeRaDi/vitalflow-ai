import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import requests
from typing import Tuple, Dict, Any, List

class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step_output = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_step_output))
        return self.fc2(x)

class DemandDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class Trainer:
    def __init__(self, db, device: torch.device):
        self.db = db
        self.device = device
        self.seq_length = 30
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout_rate = 0.2
        self.batch_size = 32
        self.epochs = 50
        self.scaler = MinMaxScaler()

    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        return np.array(sequences), np.array(targets)

    def prepare_data(self, data: List[Tuple[int, str, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        df = pd.DataFrame(data, columns=['itemid', 'date', 'total_sales'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        sales_series = df['total_sales'].values.reshape(-1, 1)
        normalized_sales = self.scaler.fit_transform(sales_series)

        X_train, y_train = self._create_sequences(normalized_sales, self.seq_length)

        return torch.tensor(X_train, dtype=torch.float32).to(self.device), \
               torch.tensor(y_train, dtype=torch.float32).to(self.device)

    def train_model(self, model: BiLSTMPredictor, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer):
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
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def _fetch_data(self, item_id: int) -> List[Tuple[int, str, float]]:
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT itemid, date, SUM(sales) AS total_sales 
                    FROM sales_warehouse 
                    WHERE itemid = %s 
                    GROUP BY itemid, date
                    ORDER BY date;
                """, (item_id,))
                return cursor.fetchall()

    def _save_model(self, model: BiLSTMPredictor, save_path: str, item_id: int):
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': self.scaler,
            'item_id': item_id,
            'training_date': datetime.now().isoformat(),
        }, save_path)

    def _upload_model(self, save_path: str, item_id: int):
        host = f"http://{os.getenv('CDN_HOST')}"
        path = f"{os.getenv('CDN_MODELS_PATH')}/lstm_model_item_{item_id}.pth"
        auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
        url = f"{host}/{path}"

        print(f"[TRAINER] Uploading model to {url}")

        with open(save_path, 'rb') as f:
            response = requests.put(url, headers={"Content-Type": "application/octet-stream"}, data=f, auth=auth)
            response.raise_for_status()

        os.remove(save_path)
        print(f"[TRAINER] Model uploaded and removed locally: {url}")

    def exec(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item_id = payload.get('item_id')
        if item_id is None:
            raise ValueError("Payload must contain 'item_id'")

        save_path = payload.get('save_path', f'lstm_model_item_{item_id}.pth')

        data = self._fetch_data(item_id)
        if not data:
            raise ValueError(f"No data found for item_id {item_id}")

        X_train, y_train = self.prepare_data(data)
        train_dataset = DemandDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        model = BiLSTMPredictor(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"[TRAINER] Starting training for item_id {item_id}")
        self.train_model(model, train_loader, criterion, optimizer)

        self._save_model(model, save_path, item_id)
        self._upload_model(save_path, item_id)

        return payload