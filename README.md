# High-Frequency Quantile LSTM Volatility Predictor

Most volatility models give you a single point prediction and leave you guessing about the tail risk. This repository implements a Quantile LSTM pipeline designed to predict the 10th, 50th, and 90th percentiles of realized volatility, giving you a full probabilistic risk band. 

We achieved a **Best RMSE of 0.0120** during the walk-forward evaluation. It is built to ingest SPY and ^VIX data, learn the market's nonlinear panic functions, and output predictions that actually respect the asymmetry of financial drawdowns.

## Core Architecture

* **The Data Engine (`data_pipeline.py`):** Ingests daily SPY and ^VIX data via `yfinance`. Calculates log returns, computes a 21-day rolling realized volatility, and outputs a clean, standardized Parquet file.
* **The Quantile LSTM (`lstm_model.py`):** A custom PyTorch LSTM designed specifically for financial time series. It utilizes dropout for regularization and outputs a 3-dimensional tensor corresponding to our target risk bands.
* **The Thunderdome (`train.py`):** The heavy lifter. Features `Optuna` for aggressive hyperparameter tuning (learning rate, hidden size, layers, dropout) using a Tree-structured Parzen Estimator (TPE). Implements strict Walk-Forward Validation across multiple eras to prevent data leakage.
* **The Asymmetric Pinball Loss:** Standard Mean Squared Error is useless for tail risk. We use a custom Pinball Loss function that heavily penalizes the model when it underestimates the 90th percentile of market volatility.
* **The Arena (`showdown.ipynb`):** A Jupyter notebook for visualizing the predictions, comparing the risk bands against actual realized volatility, and calculating final performance metrics.

## Model Performance

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Best RMSE** | `0.0120` | Out-of-sample walk-forward validation. |
| **Target Quantiles** | `0.10, 0.50, 0.90` | Captures the median expectation alongside extreme tail risks. |
| **Validation Method** | `Rolling Walk-Forward` | 4 distinct eras to simulate real-world trading amnesia. |

## Installation & Execution

You need PyTorch, Optuna, and a machine with decent compute. 

1.  **Install Dependencies:** ```bash
    pip install -r requirements.txt
    ```
2.  **Pull the Data:**
    ```bash
    python data_pipeline.py
    ```
3.  **Optimize and Train:**
    ```bash
    python train.py
    ```
4.  **Analyze the Results:**
    Open `showdown.ipynb` to visualize the volatility bands and verify the RMSE.

## Output

The pipeline generates optimized architecture parameters and outputs descaled, real-world volatility predictions across three risk quantiles, ready for strategy integration.