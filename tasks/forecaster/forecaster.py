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
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        input_size = checkpoint.get('input_size', 1)
        use_prophet = checkpoint.get('use_prophet', False)
        seasonality_features = checkpoint.get('seasonality_features', [])
        best_config = checkpoint.get('best_config', {})
        
        print(f"<!> Loading model with configuration: {best_config.get('name', 'Unknown')}")
        print(f"<!> Seasonality features: {seasonality_features}")
        print(f"<!> Using Prophet: {use_prophet}")
        
        model = BiLSTMModel(
            input_size=input_size, 
            hidden_size=128, 
            num_layers=3, 
            output_size=1,
            dropout_rate=0.2
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = checkpoint['scaler']
        last_training_date = datetime.fromisoformat(checkpoint.get('training_date'))
        
        prophet_model = checkpoint.get('prophet_model')
        
        return model, scaler, last_training_date, use_prophet, seasonality_features, prophet_model, best_config
    
    def forecast(self, model, scaler, last_training_date, use_prophet, seasonality_features, prophet_model, best_config, item_id):
        future_dates = [last_training_date + timedelta(days=i) for i in range(1, 31)]
        predictions = []
        
        if use_prophet and prophet_model and len(seasonality_features) > 0:
            print(f"<!> Using Prophet seasonality features for forecasting")
            
            recent_data = self.db.get_item_data(item_id)
            if not recent_data:
                raise ValueError(f"No recent data found for item_id {item_id}")
            
            df = pd.DataFrame(recent_data, columns=['date', 'daily_quantity_out'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').tail(30)
            
            all_dates = list(df['date']) + future_dates
            seasonality_df = self.predict_seasonality_for_dates(prophet_model, all_dates, seasonality_features)
            input_features = []

            for i, date in enumerate(df['date']):
                sales_value = df.iloc[i]['daily_quantity_out']
                season_row = seasonality_df[seasonality_df['ds'] == date]
                
                if len(season_row) > 0:
                    features = [sales_value] + [season_row.iloc[0][feat] for feat in seasonality_features]
                else:
                    features = [sales_value] + [0.0] * len(seasonality_features)
                
                input_features.append(features)
            
            input_features = np.array(input_features)
            normalized_input = scaler.transform(input_features)
            
            input_seq = normalized_input.reshape(1, 30, len(seasonality_features) + 1)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
            
            for day_idx in range(30):
                future_date = future_dates[day_idx]
                season_row = seasonality_df[seasonality_df['ds'] == future_date]

                if len(season_row) > 0:
                    season_features = [season_row.iloc[0][feat] for feat in seasonality_features]
                else:
                    season_features = [0.0] * len(seasonality_features)
                
                with torch.no_grad():
                    prediction = model(input_tensor).cpu().numpy()[0, 0]
                
                predictions.append(prediction)

                new_features = [prediction] + season_features
                normalized_new_features = scaler.transform([new_features])[0]
                input_seq = np.roll(input_seq, -1, axis=1)
                input_seq[0, -1] = normalized_new_features
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
        
        else:
            print(f"<!> Using simple LSTM forecasting (no seasonality)")
            
            # Simple LSTM forecasting without seasonality
            input_seq = np.zeros((1, self.seq_length, 1))
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
            
            for _ in range(30):
                with torch.no_grad():
                    prediction = model(input_tensor).cpu().numpy()[0, 0]
                
                predictions.append(prediction)
                
                input_seq = np.roll(input_seq, -1, axis=1)
                input_seq[0, -1, 0] = prediction
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
        
        if use_prophet and len(seasonality_features) > 0:
            # For Prophet models, pad predictions for inverse transform
            predictions_padded = np.column_stack([
                predictions,
                *[np.zeros(len(predictions)) for _ in range(len(seasonality_features))]
            ])
            predictions_orig = scaler.inverse_transform(predictions_padded)[:, 0]
        else:
            # For simple LSTM, direct inverse transform
            predictions_orig = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast': predictions_orig})
        return forecast_df
    
    def predict_seasonality_for_dates(self, prophet_model, dates, seasonality_features):
        future_df = pd.DataFrame({'ds': pd.to_datetime(dates)})
        forecast = prophet_model.predict(future_df)
        
        feature_columns = ['ds'] + seasonality_features
        return forecast[feature_columns]
    
    def exec(self, payload):
        item_id = payload['item_id']
        print(f"<!> Forecast for item {item_id}")
        model = f"prophet_bilstm_model_item_{item_id}.pth"
        
        try:
            model_path = self.download_model(model)
            model, scaler, last_transaction_date, use_prophet, seasonality_features, prophet_model, best_config = self.load_model(model_path)
            forecast_df = self.forecast(model, scaler, last_transaction_date, use_prophet, seasonality_features, prophet_model, best_config, item_id)
            os.remove(model_path)
        except Exception as e:
            print(f"<!> Error loading Prophet model, falling back to simple LSTM: {e}")
            model = f"lstm_model_item_{item_id}.pth"
            try:
                model_path = self.download_model(model)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                scaler = checkpoint['scaler']
                last_training_date = datetime.fromisoformat(checkpoint.get('training_date'))
                forecast_df = self.forecast(model, scaler, last_training_date, False, [], None, {}, item_id)
                os.remove(model_path)
            except Exception as e2:
                print(f"<!> Error loading fallback model: {e2}")
                raise e2

        forecasted_value = forecast_df['forecast'].sum()
        safetyStockCalculator = SafetyStockCalculator(self.db, item_id, 7, 30, forecasted_value)

        return safetyStockCalculator.exec()
