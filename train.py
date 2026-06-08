import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import optuna
from tqdm import tqdm
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

os.makedirs('_models', exist_ok=True)

print("Loading augmented train data and test data...")
train = pd.read_csv('dataset/train.csv') 
test = pd.read_csv('dataset/test.csv')

artifacts = {}
artifacts['roadtype_mode'] = train['RoadType'].mode()[0]
artifacts['temperature_median'] = train['Temperature'].median()

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
df_train = preprocess_data(train)
df_test = preprocess_data(test)

label_encoders = {}
cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for col in cat_cols:
    if col in df_train.columns:
        df_train[col] = df_train[col].astype(str)
        df_test[col] = df_test[col].astype(str)
        
        le = LabelEncoder()
        le.fit(pd.concat([df_train[col], df_test[col]], axis=0))
        df_train[col] = le.transform(df_train[col])
        label_encoders[col] = le

artifacts['label_encoders'] = label_encoders
joblib.dump(artifacts, 'models/artifacts.pkl')

X = df_train.drop(columns=['demand', 'Index'], errors='ignore')
y = df_train['demand']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cb  = np.zeros(len(X))

print("Training LGBM, XGBoost, and CatBoost...")

cat_features = [col for col in cat_cols if col in X.columns]

TUNE_MODEL_PARAMS = True
if TUNE_MODEL_PARAMS:
    print("Running Optuna Hyperparameter Tuning over a small validation set...")
    def objective_lgb(trial):
        X_tr_op, X_va_op, y_tr_op, y_va_op = train_test_split(X, y, test_size=0.2, random_state=42)
        params = {
            'objective': 'regression', 'metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 128),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': 500, 'random_state': 42, 'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr_op, y_tr_op, eval_set=[(X_va_op, y_va_op)], categorical_feature=cat_features, callbacks=[lgb.early_stopping(50, verbose=False)])
        return np.sqrt(mean_squared_error(y_va_op, model.predict(X_va_op)))
       
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=20)
    print("Best LightGBM params:", study_lgb.best_params)

lgb_models = []
xgb_models = []
cb_models = []

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X, y)), total=kf.n_splits, desc="Training Folds"):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        objective='regression', metric='rmse', 
        learning_rate=0.03, max_depth=8, num_leaves=63, 
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_estimators=2000
    )
    model_lgb.fit(
        X_tr, y_tr, 
        eval_set=[(X_va, y_va)], 
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    lgb_models.append(model_lgb)
    
    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric='rmse', 
        learning_rate=0.03, max_depth=8, 
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_estimators=2000
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    xgb_models.append(model_xgb)
    
    model_cb = cb.CatBoostRegressor(
        loss_function='RMSE', learning_rate=0.03, depth=8, 
        subsample=0.8, colsample_bylevel=0.8, 
        random_state=42, iterations=2000, verbose=False
    )
    model_cb.fit(
        X_tr, y_tr, 
        eval_set=(X_va, y_va), 
        cat_features=cat_features,
        early_stopping_rounds=100
    )
    oof_cb[val_idx] = model_cb.predict(X_va)
    cb_models.append(model_cb)

joblib.dump(lgb_models, 'models/lgb_models.pkl')
joblib.dump(xgb_models, 'models/xgb_models.pkl')
joblib.dump(cb_models, 'models/cb_models.pkl')

print("\n[Optuna] Optimizing final ensemble weights using Out-Of-Fold predictions...")

def objective_weights(trial):
    w_lgb = trial.suggest_float('w_lgb', 0, 1.0)
    w_xgb = trial.suggest_float('w_xgb', 0, 1.0)
    w_cb  = trial.suggest_float('w_cb', 0, 1.0)
    
    total = w_lgb + w_xgb + w_cb
    if total == 0:
        return float('inf')
        
    w_lgb /= total
    w_xgb /= total
    w_cb /= total
    
    ensemble_oof = (w_lgb * oof_lgb) + (w_xgb * oof_xgb) + (w_cb * oof_cb)
    
    return np.sqrt(mean_squared_error(y, ensemble_oof))

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective_weights, n_trials=250, show_progress_bar=True)

best_w = study.best_params
total_w = best_w['w_lgb'] + best_w['w_xgb'] + best_w['w_cb']

final_w_lgb = best_w['w_lgb'] / total_w
final_w_xgb = best_w['w_xgb'] / total_w
final_w_cb  = best_w['w_cb'] / total_w

weights = {
    'lgb': final_w_lgb,
    'xgb': final_w_xgb,
    'cb': final_w_cb
}
joblib.dump(weights, 'models/ensemble_weights.pkl')

print(f"Optimal Weights Found -> LGB: {final_w_lgb:.4f}, XGB: {final_w_xgb:.4f}, CB: {final_w_cb:.4f}")
print(f"Best OOF RMSE: {study.best_value:.5f}")
print("Training finished. Models and artifacts saved to 'models/' directory.")
