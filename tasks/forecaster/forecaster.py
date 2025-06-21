import os
import torch
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from models.bilstm_model import BiLSTMModel
from tasks.forecaster.safety_stock_calculator import SafetyStockCalculator

class Forecaster:
    def __init__(self, db, device):
        self.db = db
        self.device = device
        self.seq_length = 30
    
    def download_model(self, model):
        host = f"http://{os.getenv('CDN_HOST')}"
        path = f"{os.getenv('CDN_MODELS_PATH')}/{model}"
        auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
        url = f"{host}/{path}"
        
        response = requests.get(url, auth=auth)
        if response.status_code != 200:
            raise ValueError(f"Failed to download model {model}: {response.status_code}")
        
        with open(model, 'wb') as f:
            f.write(response.content)
        
        return model
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = checkpoint['scaler']

        last_training_date = datetime.fromisoformat(checkpoint.get('training_date'))
        
        return model, scaler, last_training_date
    
    def forecast(self, model, scaler, last_training_date):
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
        
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast': predictions})
        return forecast_df
    
    def exec(self, payload):
        item_id = payload['item_id']
        print(f"<!> Forecast for item {item_id}")
        model = f"lstm_model_item_{item_id}.pth"
        model_path = self.download_model(model)
        model, scaler, last_transaction_date = self.load_model(model_path)
        forecast_df = self.forecast(model, scaler, last_transaction_date)
        os.remove(model_path)

        forecasted_value = forecast_df['forecast'].sum()
        safetyStockCalculator = SafetyStockCalculator(self.db, item_id, 7, 30, forecasted_value)

        return safetyStockCalculator.exec()
