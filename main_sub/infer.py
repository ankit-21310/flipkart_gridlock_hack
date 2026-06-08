import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading test data and artifacts...")
test = pd.read_csv('dataset/test.csv')
artifacts = joblib.load('models/artifacts.pkl')
label_encoders = artifacts['label_encoders']

def preprocess_data(df):
    df = df.copy()
    
    if 'timestamp' in df.columns:
        df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(float)
        df.drop(columns=['timestamp'], inplace=True, errors='ignore')
        
        time_in_mins = df['hour'] * 60 + df['minute']
        df['sin_time'] = np.sin(2 * np.pi * time_in_mins / 1440)
        df['cos_time'] = np.cos(2 * np.pi * time_in_mins / 1440)
        
    if 'RoadType' in df.columns:
        df['RoadType'] = df['RoadType'].fillna(artifacts['roadtype_mode'])
    
    if 'hour' in df.columns:
        if 'geohash' in df.columns:
            df['Temperature'] = df.groupby(['geohash', 'hour'])['Temperature'].transform(lambda x: x.fillna(x.median()))
            
        df['Temperature'] = df.groupby(['day', 'hour'])['Temperature'].transform(lambda x: x.fillna(x.median()))
        df['Temperature'] = df['Temperature'].fillna(artifacts['temperature_median'])
        
        if 'geohash' in df.columns:
            df['Weather'] = df.groupby(['geohash'])['Weather'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
            
        df['Weather'] = df.groupby(['day'])['Weather'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
        df['Weather'] = df['Weather'].fillna('Sunny')
        
    return df

print("Preprocessing data...")
df_test = preprocess_data(test)

cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for col in cat_cols:
    if col in df_test.columns and col in label_encoders:
        le = label_encoders[col]
        # In case of unseen labels, fallback to -1 or a known class, or just transform since we fitted on both.
        # It's assumed test data objects are fully known to the le from train.py's fit process.
        df_test[col] = df_test[col].astype(str)
        df_test[col] = le.transform(df_test[col])

X_test = df_test.drop(columns=['demand', 'Index'], errors='ignore')
test_idx = df_test['Index']

print("Loading models...")
lgb_models = joblib.load('models/lgb_models.pkl')
xgb_models = joblib.load('models/xgb_models.pkl')
cb_models = joblib.load('models/cb_models.pkl')
weights = joblib.load('models/ensemble_weights.pkl')

print("Generating predictions...")
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cb  = np.zeros(len(X_test))

for model in lgb_models:
    test_preds_lgb += model.predict(X_test) / len(lgb_models)
    
for model in xgb_models:
    test_preds_xgb += model.predict(X_test) / len(xgb_models)
    
for model in cb_models:
    test_preds_cb += model.predict(X_test) / len(cb_models)

test_preds_ensemble = (weights['lgb'] * test_preds_lgb) + (weights['xgb'] * test_preds_xgb) + (weights['cb'] * test_preds_cb)

sub = pd.DataFrame({'Index': test_idx, 'demand': test_preds_ensemble})
sub.to_csv('submission_final.csv', index=False)
print("\n✅ Submission generated successfully as 'submission_final.csv'")
