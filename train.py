import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import optuna
from lstm_model import QuantileLSTM

torch.manual_seed(42)
np.random.seed(42)

class PinballLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds shape: (Batch, 3) -> The three risk bands
        # target shape: (Batch, 1) -> The actual real-world volatility
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i:i+1]
            loss += torch.max(q * error, (q-1) * error).mean()
        return loss

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, 1])
    return np.array(xs), np.array(ys)

def run_optimized_pipeline(
        file_path="SPY_VIX_daily_clean.parquet", 
        n_trials=10, 
        n_splits=4
    ):
    df = pd.read_parquet(file_path)
    seq_length = 21
    
    test_start_idx = int(len(df) * 0.8)
    optuna_train_df = df.iloc[:test_start_idx]
    test_df = df.iloc[test_start_idx:]

    print("1. Prepping Tensor Data for Optuna...")
    scaler = StandardScaler()
    # THE UPGRADE: 6 Features instead of 3
    features = [
        'log_return',
        'realized_vol',
        'vix_close'
    ]
    train_scaled = scaler.fit_transform(optuna_train_df[features])
    
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    print(f"2. The Thunderdome (Running {n_trials} Trials to find Architecture)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.15, 0.35)
        lr = trial.suggest_float('lr', 0.001, 0.008)
        
        model = QuantileLSTM(
            input_size=3, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])
        
        for _ in range(40): 
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
        return loss.item()

    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    print(f"--> Architecture Locked: {best}")

    print(f"3. Executing Rolling Window Walk-Forward Validation ({n_splits} Eras)...")
    step_size = len(test_df) // n_splits
    all_descaled_preds = []
    
    train_window_size = test_start_idx
    
    for step in range(n_splits):
        print(f"   -> Inducing Amnesia and Training Era {step + 1}/{n_splits}...")
        current_train_end = test_start_idx + (step * step_size)
        current_test_end = current_train_end + step_size if step < n_splits - 1 else len(df)
        
        window_start = current_train_end - train_window_size
        fold_train_df = df.iloc[window_start:current_train_end]
        
        fold_test_df = df.iloc[current_train_end - seq_length:current_test_end]
        
        fold_scaler = StandardScaler()
        fold_train_scaled = fold_scaler.fit_transform(fold_train_df[features])
        fold_test_scaled = fold_scaler.transform(fold_test_df[features])
        
        f_X_train, f_y_train = create_sequences(fold_train_scaled, seq_length)
        f_X_test, _ = create_sequences(fold_test_scaled, seq_length)
        
        f_X_train_t = torch.tensor(f_X_train, dtype=torch.float32)
        f_y_train_t = torch.tensor(f_y_train, dtype=torch.float32).unsqueeze(1)
        f_X_test_t = torch.tensor(f_X_test, dtype=torch.float32)
        
        torch.manual_seed(42)
        fold_model = QuantileLSTM(
            input_size=3, 
            hidden_size=best['hidden_size'], 
            num_layers=best['num_layers'], 
            dropout=best['dropout']
        )
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=best['lr'])
        criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])
        
        for epoch in range(50):
            fold_model.train()
            fold_optimizer.zero_grad()
            loss = criterion(fold_model(f_X_train_t), f_y_train_t)
            loss.backward()
            fold_optimizer.step()
            
        fold_model.eval()
        with torch.no_grad():
            preds_scaled = fold_model(f_X_test_t).numpy()

        dummy_10 = np.zeros((len(preds_scaled), 3))
        dummy_50 = np.zeros((len(preds_scaled), 3))
        dummy_90 = np.zeros((len(preds_scaled), 3))

        dummy_10[:, 1] = preds_scaled[:, 0] # 10th
        dummy_50[:, 1] = preds_scaled[:, 1] # 50th
        dummy_90[:, 1] = preds_scaled[:, 2] # 90th

        p10 = fold_scaler.inverse_transform(dummy_10)[:, 1]
        p50 = fold_scaler.inverse_transform(dummy_50)[:, 1]
        p90 = fold_scaler.inverse_transform(dummy_90)[:, 1]

        preds_descaled = np.column_stack((p10, p50, p90))
        all_descaled_preds.extend(preds_descaled)

    final_wf_predictions = np.array(all_descaled_preds)
    return test_df, final_wf_predictions, best, fold_model, fold_scaler, features