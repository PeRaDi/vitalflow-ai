import os
import torch
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, List
from models.bilstm_model import BiLSTMPredictor

class Forecaster:
    def __init__(self, device: torch.device):
        self.device = device
        self.seq_length = 30
    
    def _download_model(self, model_name: str) -> str:
        host = f"http://{os.getenv('CDN_HOST')}"
        path = f"{os.getenv('CDN_MODELS_PATH')}/{model_name}"
        auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
        url = f"{host}/{path}"
        
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        
        model_path = f"./{model_name}"
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        return model_path
    
    def _load_model(self, model_path: str) -> Tuple[BiLSTMPredictor, MinMaxScaler, datetime]:
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = BiLSTMPredictor(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler: MinMaxScaler = checkpoint['scaler']
        last_training_date = datetime.fromisoformat(checkpoint['training_date'])
        
        return model, scaler, last_training_date
    
    def _generate_forecast(self, model: BiLSTMPredictor, scaler: MinMaxScaler, last_training_date: datetime) -> pd.DataFrame:
        future_dates = [last_training_date + timedelta(days=i) for i in range(1, 31)]
        predictions = []
        
        input_seq = np.zeros((1, self.seq_length, 1))
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
        
        for _ in range(30):
            with torch.no_grad():
                prediction = model(input_tensor).cpu().numpy()[0, 0]
            
            predictions.append(prediction)
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 0] = prediction
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        return pd.DataFrame({'date': future_dates, 'predicted_sales': predictions})
    
    def exec(self, payload: Dict[str, Any]) -> float:
        item_id = payload.get('item_id')
        if item_id is None:
            raise ValueError("Payload must contain 'item_id'")
        
        print(f"[FORECASTER] Forecasting for item {item_id}")
        model_name = f"lstm_model_item_{item_id}.pth"
        model_path = self._download_model(model_name)
        
        model, scaler, last_training_date = self._load_model(model_path)
        forecast_df = self._generate_forecast(model, scaler, last_training_date)
        
        os.remove(model_path)
        
        return float(forecast_df['predicted_sales'].sum())
