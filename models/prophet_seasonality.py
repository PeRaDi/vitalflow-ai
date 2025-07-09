import pandas as pd
from prophet import Prophet


class ProphetSeasonalityExtractor:    
    def __init__(self):
        self.prophet_model = None
        self.config = None
        
    def get_seasonality_combinations(self):
        return [
            {'features': [], 'name': 'No_Seasonality', 'weekly': False, 'yearly': False, 'monthly': False},
            # {'features': ['weekly'], 'name': 'Weekly_Only', 'weekly': True, 'yearly': False, 'monthly': False},
            # {'features': ['yearly'], 'name': 'Yearly_Only', 'weekly': False, 'yearly': True, 'monthly': False},
            # {'features': ['monthly'], 'name': 'Monthly_Only', 'weekly': False, 'yearly': False, 'monthly': True},
            # {'features': ['weekly', 'yearly'], 'name': 'Weekly_Yearly', 'weekly': True, 'yearly': True, 'monthly': False},
            # {'features': ['yearly', 'monthly'], 'name': 'Yearly_Monthly', 'weekly': False, 'yearly': True, 'monthly': True},
            # {'features': ['weekly', 'monthly'], 'name': 'Weekly_Monthly', 'weekly': True, 'yearly': False, 'monthly': True},
            # {'features': ['weekly', 'yearly', 'monthly'], 'name': 'All_Seasonality', 'weekly': True, 'yearly': True, 'monthly': True},
        ]
    
    def preprocess_for_prophet(self, data):
        df = pd.DataFrame(data, columns=['date', 'quantity'])
        df['ds'] = pd.to_datetime(df['date'])
        df = df.groupby('ds')['quantity'].sum().reset_index()
        df = df.rename(columns={'quantity': 'y'})
        return df
    
    def check_data_volume(self, df):
        if len(df) < 2:
            return False, "Not enough data points"
        
        min_date = df['ds'].min()
        max_date = df['ds'].max()
        years_diff = (max_date - min_date).days / 365.25
        
        if years_diff >= 2.0:
            return True, f"Sufficient data: {years_diff:.2f} years"
        else:
            return False, f"Insufficient data: {years_diff:.2f} years (need 2+ years)"
    
    def train_prophet_model(self, prophet_df, config):
        model = Prophet(
            yearly_seasonality=config['yearly'],
            weekly_seasonality=config['weekly'],
            daily_seasonality=False
        )
        
        if config['monthly']:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(prophet_df)
        self.prophet_model = model
        self.config = config
        
        return model
    
    def extract_seasonality_features(self, prophet_df, config, future_days=30):
        model = self.train_prophet_model(prophet_df, config)
        
        future = model.make_future_dataframe(periods=future_days)
        forecast = model.predict(future)
    
        feature_columns = ['ds']
        if config['weekly']:
            feature_columns.append('weekly')
        if config['yearly']:
            feature_columns.append('yearly')
        if config['monthly']:
            feature_columns.append('monthly')
            
        return forecast[feature_columns]
    
    def get_fallback_features(self, df):
        df_copy = df.copy()
        
        df_copy['day_of_month'] = df_copy['ds'].dt.day
        df_copy['month'] = df_copy['ds'].dt.month
        df_copy['monthly_simple'] = (df_copy['day_of_month'] - 15.5) / 15.5
        df_copy['yearly_simple'] = (df_copy['month'] - 6.5) / 6.5
        
        return df_copy[['ds', 'yearly_simple', 'monthly_simple']]
    
    def predict_seasonality(self, future_dates):
        if not self.prophet_model or not self.config:
            raise ValueError("Prophet model not trained. Call extract_seasonality_features first.")
        
        future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
        forecast = self.prophet_model.predict(future_df)
        
        feature_columns = ['ds']
        if self.config['weekly']:
            feature_columns.append('weekly')
        if self.config['yearly']:
            feature_columns.append('yearly')
        if self.config['monthly']:
            feature_columns.append('monthly')
            
        return forecast[feature_columns]
