import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from datetime import datetime
import requests
import json
from models.bilstm_model import BiLSTMModel
from models.demand_dataset import DemandDataset
from models.prophet_seasonality import ProphetSeasonalityExtractor

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
        self.prophet_extractor = ProphetSeasonalityExtractor()
        self.best_model_config = None
        self.seasonality_features = []
        self.use_prophet = False
        
    def create_sequences(self, data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        return np.array(sequences), np.array(targets)

    def prepare_data_with_config(self, data, config, test_split=0.2):
        """Prepare data with specific seasonality configuration"""
        df = pd.DataFrame(data, columns=['date', 'daily_quantity_out'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Rename column for consistency
        df = df.rename(columns={'daily_quantity_out': 'total_sales'})
        
        prophet_df = self.prophet_extractor.preprocess_for_prophet([(row['date'], row['total_sales']) for _, row in df.iterrows()])
        
        if len(config['features']) == 0:
            # No seasonality - just use sales data
            features = df[['total_sales']].values
            feature_names = []
        else:
            # Use Prophet with specified seasonality
            seasonality_df = self.prophet_extractor.extract_seasonality_features(prophet_df, config)
            
            # Merge seasonality features
            merge_columns = ['ds'] + config['features']
            df = df.merge(seasonality_df[merge_columns], left_on='date', right_on='ds', how='left')
            df = df.fillna(0)
            
            # Prepare feature matrix
            feature_columns = ['total_sales'] + config['features']
            features = df[feature_columns].values
            feature_names = config['features']
        
        # Normalize features
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(normalized_features, self.seq_length)
        
        # Train/test split
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train[:, 0:1], dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test[:, 0:1], dtype=torch.float32).to(self.device)
        
        return X_train, y_train, X_test, y_test, scaler, feature_names

    def exec(self, payload):
        item_id = payload['item_id']
        print("-" * 50)
        print(f"<!> TRAINING ITEM_ID: {item_id}")
        save_path = payload.get('save_path', f'prophet_bilstm_model_item_{item_id}.pth')
        
        data = self.db.get_item_data(item_id)
        if not data:
            raise ValueError(f"No data found for item_id {item_id}")

        # Check if we have enough data for Prophet (2+ years)
        prophet_data = [(row[0], row[1]) for row in data]  # Convert to tuple format
        prophet_df = self.prophet_extractor.preprocess_for_prophet(prophet_data)
        has_enough_data, message = self.prophet_extractor.check_data_volume(prophet_df)
        
        if has_enough_data:
            print(f"  - Prophet? : Yes")
            
            # Test all seasonality combinations
            all_results = []
            seasonality_combinations = self.prophet_extractor.get_seasonality_combinations()
            
            for config in seasonality_combinations:
                print(f"  - Testing configuration: {config['features']}")
                result = self.train_and_evaluate_config(data, config)
                if result:
                    all_results.append(result)
            
            if not all_results:
                raise ValueError("No valid configurations could be trained")
            
            # Select best model based on RMSE (lower is better)
            best_result = min(all_results, key=lambda x: x['metrics']['rmse'])
            
            # Store best model configuration
            self.best_model_config = best_result['config']
            self.use_prophet = len(best_result['config']['features']) > 0
            self.seasonality_features = best_result['config']['features']

            # Use best model and its results
            model = best_result['model']
            scaler = best_result['scaler']
            test_metrics = best_result['metrics']
            
            # Save comparison results
            comparison_report = {
                'item_id': item_id,
                'timestamp': datetime.now().isoformat(),
                'data_years': (prophet_df['ds'].max() - prophet_df['ds'].min()).days / 365.25,
                'best_config': best_result['config']['name'],
                'best_features': best_result['config']['features'],
                'best_metrics': best_result['metrics'],
                'all_results': [
                    {
                        'config': r['config']['name'],
                        'features': r['config']['features'],
                        'metrics': r['metrics']
                    }
                    for r in all_results
                ]
            }
            
            comparison_path = f'model_comparison_item_{item_id}.json'
            with open(comparison_path, 'w') as f:
                json.dump(comparison_report, f, indent=2)
            
        else:
            print(f"  - Prophet? : No")
            
            # Fallback to simple LSTM without seasonality
            fallback_config = {'features': [], 'name': 'Simple_LSTM_Fallback', 'weekly': False, 'yearly': False, 'monthly': False}
            result = self.train_and_evaluate_config(data, fallback_config)
            
            if not result:
                raise ValueError("Fallback configuration failed")
            
            # Use fallback results
            model = result['model']
            scaler = result['scaler']
            test_metrics = result['metrics']
            
            self.best_model_config = fallback_config
            self.use_prophet = False
            self.seasonality_features = []
        
        # Calculate training metrics
        train_metrics = self.calculate_training_metrics_from_model(model, scaler, data)
        
        # Combine all metrics
        all_metrics = {**test_metrics, **train_metrics}
        
        # Save evaluation report
        report_path = self.save_evaluation_report(item_id, all_metrics, 
                                                 best_result['predictions'] if has_enough_data else result['predictions'],
                                                 best_result['targets'] if has_enough_data else result['targets'])
        
        # Print evaluation results
        print(f"  - Model Evaluation Results for item_id {item_id}:")
        print(f"     - Best Configuration: {self.best_model_config['name']}")
        print(f"     - Seasonality Features: {self.seasonality_features}")
        print(f"     - Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"     - Test MAE: {test_metrics['mae']:.4f}")
        print(f"     - Test R²: {test_metrics['r2_score']:.4f}")
        print(f"     - Test MAPE: {test_metrics['mape']:.2f}%")
        print(f"     - Directional Accuracy: {test_metrics['directional_accuracy']:.2f}%")
        print(f"     - Train RMSE: {train_metrics['train_rmse']:.4f}")
        print(f"     - Train R²: {train_metrics['train_r2']:.4f}")

        # Save model with comprehensive metadata
        model_data = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'item_id': item_id,
            'training_date': datetime.now().isoformat(),
            'evaluation_metrics': all_metrics,
            'use_prophet': self.use_prophet,
            'seasonality_features': self.seasonality_features,
            'input_size': len(self.seasonality_features) + 1,
            'approach': f"Adaptive_{self.best_model_config['name']}",
            'best_config': self.best_model_config,
            'data_years': (prophet_df['ds'].max() - prophet_df['ds'].min()).days / 365.25
        }
        
        # Add Prophet model only if used
        if self.use_prophet and self.prophet_extractor.prophet_model:
            model_data['prophet_model'] = self.prophet_extractor.prophet_model
            
        torch.save(model_data, save_path)
        
        # Upload to CDN
        host = f"http://{os.getenv('CDN_HOST')}"
        path = f"{os.getenv('CDN_MODELS_PATH')}/prophet_bilstm_model_item_{item_id}.pth"
        auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
        url = f"{host}/{path}"

        print(f"  - Uploading model to {host}/{path}")
        with open(save_path, 'rb') as f:
            headers = {"Content-Type": "application/octet-stream"}
            response = requests.put(url, headers=headers, data=f, auth=auth)
            response.raise_for_status()
            f.close()
        
        os.remove(save_path)
        print(f"  - Done")
        
        # Add evaluation metrics to payload
        payload['evaluation_metrics'] = all_metrics
        payload['evaluation_report_path'] = report_path
        payload['model_performance'] = {
            'best_config': self.best_model_config['name'],
            'seasonality_features': self.seasonality_features,
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2_score'],
            'test_mape': test_metrics['mape'],
            'directional_accuracy': test_metrics['directional_accuracy'],
            'train_rmse': train_metrics['train_rmse'],
            'train_r2': train_metrics['train_r2'],
            'use_prophet': self.use_prophet,
            'input_size': model_data['input_size'],
            'approach': model_data['approach']
        }

        return payload

    def save_evaluation_report(self, item_id, metrics, predictions, targets):
        """
        Save detailed evaluation report as JSON for further analysis
        """
        
        report = {
            'item_id': item_id,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_config': {
                'seq_length': self.seq_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
            'predictions_sample': predictions[:10].tolist() if len(predictions) > 10 else predictions.tolist(),
            'targets_sample': targets[:10].tolist() if len(targets) > 10 else targets.tolist(),
            'prediction_statistics': {
                'mean_prediction': float(np.mean(predictions)),
                'std_prediction': float(np.std(predictions)),
                'min_prediction': float(np.min(predictions)),
                'max_prediction': float(np.max(predictions)),
                'mean_target': float(np.mean(targets)),
                'std_target': float(np.std(targets)),
                'min_target': float(np.min(targets)),
                'max_target': float(np.max(targets))
            }
        }
        
        report_path = f'evaluation_report_item_{item_id}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        return report_path

    def train_and_evaluate_config(self, data, config):
        """Train and evaluate a model with specific seasonality configuration"""
        try:
            X_train, y_train, X_test, y_test, scaler, feature_names = self.prepare_data_with_config(data, config)
            
            # Create data loaders
            train_dataset = DemandDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            test_dataset = DemandDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            input_size = len(config['features']) + 1  # +1 for sales
            model = BiLSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model (reduced epochs for faster comparison)
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
                
                if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                    print(f"      Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
            
            # Evaluate model
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = model(X_batch)
                    all_predictions.extend(y_pred.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())
            
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            
            # Inverse transform
            feature_count = input_size
            predictions_padded = np.column_stack([
                predictions.flatten(),
                *[np.zeros(len(predictions)) for _ in range(feature_count - 1)]
            ])
            targets_padded = np.column_stack([
                targets.flatten(),
                *[np.zeros(len(targets)) for _ in range(feature_count - 1)]
            ])
            
            predictions_orig = scaler.inverse_transform(predictions_padded)[:, 0]
            targets_orig = scaler.inverse_transform(targets_padded)[:, 0]
            
            # Calculate metrics
            mse = mean_squared_error(targets_orig, predictions_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets_orig, predictions_orig)
            r2 = r2_score(targets_orig, predictions_orig)
            mape = np.mean(np.abs((targets_orig - predictions_orig) / targets_orig)) * 100
            
            # Calculate directional accuracy
            target_direction = np.diff(targets_orig) > 0
            pred_direction = np.diff(predictions_orig) > 0
            directional_accuracy = np.mean(target_direction == pred_direction) * 100
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'test_samples': len(targets_orig)
            }
            
            return {
                'config': config,
                'metrics': metrics,
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'predictions': predictions_orig,
                'targets': targets_orig
            }
            
        except Exception as e:
            print(f"    Error with configuration {config['name']}: {e}")
            return None

    def calculate_training_metrics_from_model(self, model, scaler, data):
        """Calculate training metrics from the trained model"""
        # Prepare data again for training metrics
        current_config = {
            'features': self.seasonality_features, 
            'name': 'Current_Config', 
            'weekly': 'weekly' in self.seasonality_features,
            'yearly': 'yearly' in self.seasonality_features,
            'monthly': 'monthly' in self.seasonality_features
        }
        X_train, y_train, _, _, _, _ = self.prepare_data_with_config(data, current_config)
        
        # Create data loader
        train_dataset = DemandDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = model(X_batch)
                all_predictions.extend(y_pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Inverse transform to get original scale
        feature_count = len(self.seasonality_features) + 1
        predictions_padded = np.column_stack([
            predictions.flatten(),
            *[np.zeros(len(predictions)) for _ in range(feature_count - 1)]
        ])
        targets_padded = np.column_stack([
            targets.flatten(),
            *[np.zeros(len(targets)) for _ in range(feature_count - 1)]
        ])
        
        predictions_orig = scaler.inverse_transform(predictions_padded)[:, 0]
        targets_orig = scaler.inverse_transform(targets_padded)[:, 0]
        
        # Calculate basic metrics
        mse = mean_squared_error(targets_orig, predictions_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_orig, predictions_orig)
        r2 = r2_score(targets_orig, predictions_orig)
        
        return {
            'train_mse': float(mse),
            'train_rmse': float(rmse),
            'train_mae': float(mae),
            'train_r2': float(r2),
            'train_samples': len(targets_orig)
        }