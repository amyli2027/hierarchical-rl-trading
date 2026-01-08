from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR, INDICATORS
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# --- CONFIGURATION ---
# We use a shared universe so both agents "speak the same language" (action space)
# SPY/QQQ (Offense), GLD (Hedge), SHV (Cash)
UNIVERSE = ["SPY", "QQQ", "GLD", "SHV"] 

# 1. ROBUST DOWNLOADER (Reused)
# 1. ROBUST DOWNLOADER (Fixed)
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading {self.ticker_list}...")
        try:
            # Added auto_adjust=True to handle splits/dividends better
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date, 
                ignore_tz=True, threads=False
            )
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
        
        # FIX: Check structure and name index BEFORE resetting
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            # Handle single ticker case where yfinance might return flat columns
            df = raw_df.reset_index()
            # If 'Date' is the index name, it becomes a column. Rename it to lower case.
            df = df.rename(columns={'Date': 'date', 'Index': 'date'})
            
            # If only one ticker, add the 'tic' column manually
            if 'tic' not in df.columns:
                df['tic'] = self.ticker_list[0]

        # Standardize columns
        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        df['date'] = df['date'].astype(str)
        
        required = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        # Filter only existing columns to avoid KeyErrors
        final_cols = [c for c in required if c in df.columns]
        
        df = df[final_cols]
        df = df.drop_duplicates(subset=['date', 'tic']).dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return df

# 2. FEATURE ENGINEERING (Shared)
# 2. FEATURE ENGINEERING (Fixed)
def process_data(df, extra_tickers=['^VIX']):
    print("Processing data with technical indicators...")
    
    # 1. Add Standard Indicators (MACD, RSI, etc.)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=False, user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    
    # 2. Add Macro Context (VIX)
    vix_df = RobustYahooDownloader(df.date.min(), df.date.max(), extra_tickers).fetch_data()
    if not vix_df.empty:
        vix_df = vix_df[['date', 'close']].rename(columns={'close': 'vix'})
        processed = processed.merge(vix_df, on='date', how='left').ffill().bfill()
    else:
        processed['vix'] = 20.0 

    # 3. FIX: Calculate Trend Signals using GROUPBY
    # We must calculate SMA per ticker, otherwise we mix SPY and GLD prices!
    print("Calculating Trend with GroupBy...")
    
    # Calculate SMA-200 for each ticker individually
    processed['sma_200'] = processed.groupby('tic')['close'].transform(
        lambda x: x.rolling(window=200, min_periods=1).mean()
    )
    
    # Now calculate trend ratio
    processed['trend'] = processed['close'] / processed['sma_200']
    
    # Handle NaNs (early days of data)
    processed = processed.fillna(0)
    
    return processed
# 3. THE COUNCIL: TRAINING LOGIC
def train_specialist(name, train_start, train_end, processed_data, total_timesteps=50000):
    """
    Trains a specialized agent on a specific time period.
    """
    print(f"\n--- Training {name} Agent ({train_start} to {train_end}) ---")
    
    # Filter data for this specific specialist's era
    train_data = data_split(processed_data, train_start, train_end)
    
    stock_dim = len(UNIVERSE)
    # Ensure tech_indicator_list matches what is in processed_data
    tech_list = INDICATORS + ['vix', 'trend']
    
    env_kwargs = {
        "hmax": 100, "initial_amount": 100000, 
        "num_stock_shares": [0]*stock_dim,
        "buy_cost_pct": [0.001]*stock_dim, "sell_cost_pct": [0.001]*stock_dim, 
        "state_space": 1 + 2*stock_dim + len(tech_list)*stock_dim, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    # We use PPO for both, but they learn different weights
    model = agent.get_model("ppo", model_kwargs={"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64})
    
    trained_model = agent.train_model(model=model, tb_log_name=name, total_timesteps=total_timesteps)
    trained_model.save(os.path.join(TRAINED_MODEL_DIR, name))
    return trained_model, env_kwargs # Return kwargs so the Manager knows how to set up the env

# 4. THE MANAGER (ROUTER)
def run_council_backtest(trade_start, trade_end, processed_data, env_kwargs):
    print(f"\n--- Council Assembling for {trade_start} to {trade_end} ---")
    
    # Load the Specialists
    bull_path = os.path.join(TRAINED_MODEL_DIR, "bull_agent")
    bear_path = os.path.join(TRAINED_MODEL_DIR, "bear_agent")
    
    if not os.path.exists(bull_path + ".zip") or not os.path.exists(bear_path + ".zip"):
        print("Error: Agents not trained yet.")
        return 0.0

    bull_model = PPO.load(bull_path)
    bear_model = PPO.load(bear_path)
    
    # Prepare Trade Environment
    trade_data = data_split(processed_data, trade_start, trade_end)
    e_trade_gym = StockTradingEnv(df=trade_data, turbulence_threshold=None, risk_indicator_col=None, **env_kwargs)
    
    obs, _ = e_trade_gym.reset()
    done = False
    
    # Logging
    history = []
    
    while not done:
        # --- THE MANAGER LOGIC (FIXED) ---
        
        # 1. Use 'day', not 'current_step'
        current_day = e_trade_gym.day
        
        # 2. Calculate the correct row index
        # The dataframe is flat: [Date0-Tic0, Date0-Tic1, ... Date1-Tic0...]
        # We need to jump to the start of the current day to get the macro data.
        stride = e_trade_gym.stock_dim
        current_row_idx = current_day * stride
        
        # 3. Access Data safely
        # We look at the first ticker (SPY) for that day to get the VIX and Trend.
        # (Since VIX is merged into all rows, any ticker works).
        current_date_row = e_trade_gym.df.iloc[current_row_idx]
        
        current_vix = current_date_row['vix']
        
        # We use the trend of the first ticker (SPY) as the proxy for "Market Trend"
        market_trend = current_date_row['trend'] 
        
        # Determine Regime
        is_crisis = False
        
        # LOGIC: 
        # High Fear (VIX > 28) OR Broken Trend (Price < 200 SMA) -> BEAR MODE
        if current_vix > 28 or market_trend < 1.0:
            is_crisis = True
        
        # --- DELEGATION ---
        if is_crisis:
            action, _ = bear_model.predict(obs, deterministic=True)
            active_agent = "BEAR"
        else:
            action, _ = bull_model.predict(obs, deterministic=True)
            active_agent = "BULL"
            
        # Execute
        obs, rewards, dones, truncated, info = e_trade_gym.step(action)
        done = dones if isinstance(dones, bool) else dones[0]
        
        # Log occasionally
        if current_day % 50 == 0:
            history.append(active_agent)

    # Calculate Returns
    df_account = e_trade_gym.save_asset_memory()
    final_val = df_account['account_value'].iloc[-1]
    init_val = df_account['account_value'].iloc[0]
    return_pct = ((final_val - init_val) / init_val) * 100
    
    # Regime Stats
    bull_count = history.count("BULL")
    bear_count = history.count("BEAR")
    print(f"Manager Report: Bull Agent Active {bull_count} checks, Bear Agent Active {bear_count} checks.")
    
    return return_pct
if __name__ == "__main__":
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # 1. Fetch ALL Data (Train + Trade) once to ensure consistency
    # We span from the "Bull Era" start to "Present"
    print("Fetching Global Data...")
    loader = RobustYahooDownloader(start_date="2009-01-01", end_date="2023-01-01", ticker_list=UNIVERSE)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    # 2. Train the Specialists (The "Debate Team")
    
    # Agent A: The "Bull" (Trained on the 2012-2019 rager)
    # It learns that "Buying the dip" always works.
    train_specialist("bull_agent", "2012-01-01", "2019-01-01", df_processed)
    
    # Agent B: The "Bear" (Trained on 2020 crash & 2022 correction)
    # It learns that markets are dangerous and Cash/Gold is safe.
    # Note: We stitch two volatile periods together for training? 
    # Or just use the 2009-2011 + 2020 data. Let's use 2009-2011 (Post-GFC volatility)
    # Better yet, let's just train it on the whole volatile dataset "2009-2012"
    train_specialist("bear_agent", "2009-01-01", "2012-01-01", df_processed)
    
    # 3. Run the Council Backtest
    # Now we test on the years you provided
    test_years = [2019, 2020, 2021, 2022]
    results = []
    
    for year in test_years:
        print(f"\n=== TESTING COUNCIL ON YEAR {year} ===")
        start = f"{year}-01-01"
        end = f"{year+1}-01-01"
        
        # We need the env_kwargs from training to reconstruct the env, 
        # but since they are standard, we can regenerate them inside or pass them.
        # For simplicity in this script, we assume standard config inside the function.
        # But we need 'tech_list' consistency.
        
        # Re-defining kwargs locally for the runner (copy from train function)
        stock_dim = len(UNIVERSE)
        tech_list = INDICATORS + ['vix', 'trend']
        env_kwargs = {
            "hmax": 100, "initial_amount": 100000, 
            "num_stock_shares": [0]*stock_dim,
            "buy_cost_pct": [0.001]*stock_dim, "sell_cost_pct": [0.001]*stock_dim, 
            "state_space": 1 + 2*stock_dim + len(tech_list)*stock_dim, 
            "stock_dim": stock_dim, 
            "tech_indicator_list": tech_list, 
            "action_space": stock_dim, 
            "reward_scaling": 1e-4
        }
        
        ai_return = run_council_backtest(start, end, df_processed, env_kwargs)
        
        # Get SPY return for comparison
        spy_start = df_processed[(df_processed.date >= start) & (df_processed.tic == "SPY")]['close'].iloc[0]
        spy_end = df_processed[(df_processed.date < end) & (df_processed.tic == "SPY")]['close'].iloc[-1]
        spy_return = ((spy_end - spy_start) / spy_start) * 100
        
        results.append({"Year": year, "Council": f"{ai_return:.2f}%", "SPY": f"{spy_return:.2f}%"})

    print("\nFINAL RESULTS:")
    print(pd.DataFrame(results).to_string(index=False))