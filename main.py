from __future__ import annotations
import itertools
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, INDICATORS, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# --- 1. ROBUST DOWNLOADER ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        try:
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date,
                ignore_tz=True, threads=False 
            )
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()
            if 'tic' not in df.columns:
                df['tic'] = self.ticker_list[0]

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        df = df.drop_duplicates(subset=['date', 'tic']).dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        return df

def train_ensemble_agent(env_train, model_name, seed):
    """Trains a single agent with a specific random seed."""
    model_ppo = PPO(
        "MlpPolicy", 
        env_train, 
        n_steps=2048, 
        ent_coef=0.005, 
        learning_rate=0.00025, 
        batch_size=128, 
        gamma=0.999, 
        seed=seed,
        verbose=0
    )
    
    new_logger = configure(os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}"), ["csv"])
    model_ppo.set_logger(new_logger)
    
    print(f"   > Training Agent (Seed {seed})...")
    model_ppo.learn(total_timesteps=60000, tb_log_name=f"{model_name}_{seed}")
    return model_ppo

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float:
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # --- ASSETS: GLOBAL MACRO + TECH ---
    TARGET_ASSETS = ["SPY", "GLD", "XLE", "SHV"]
    
    print(f"Fetching Assets: {TARGET_ASSETS}")
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=TARGET_ASSETS
    ).fetch_data()
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=False, user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # --- MACRO SIGNALS ---
    print(f"Fetching Signals (^VIX, ^TNX)...")
    signal_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=["^VIX", "^TNX"]
    ).fetch_data()
    
    signal_pivot = signal_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {}
    if '^VIX' in signal_pivot.columns: rename_map['^VIX'] = 'vix'
    if '^TNX' in signal_pivot.columns: rename_map['^TNX'] = 'rate'
    signal_pivot = signal_pivot.rename(columns=rename_map)
    
    processed = processed.merge(signal_pivot, on='date', how='left')
    if 'vix' in processed.columns: processed['vix'] = processed['vix'].ffill().bfill()
    if 'rate' in processed.columns: processed['rate'] = processed['rate'].ffill().bfill()
    
    # --- FEATURE ENGINEERING ---
    # 1. VIX Signals
    processed['vix_ma'] = processed['vix'].rolling(10).mean()
    processed['fear_signal'] = processed['vix'] / processed['vix_ma']
    
    # 2. Rate Signals
    processed['rate_velocity'] = processed['rate'].pct_change(5).fillna(0)
    
    # 3. NEW: RELATIVE VOLUME (RVOL)
    # 1.0 = Normal, 2.0 = High Activity, 0.5 = Low Activity
    processed['vol_ma'] = processed['volume'].rolling(20).mean()
    processed['rel_volume'] = processed['volume'] / processed['vol_ma']
    processed['rel_volume'] = processed['rel_volume'].fillna(1.0)

    processed = processed.fillna(1.0)
    
    # Scaling
    for tic in processed['tic'].unique():
        mask = processed['tic'] == tic
        if mask.any():
            start_price = processed.loc[mask, 'close'].iloc[0]
            processed.loc[mask, ['open', 'high', 'low', 'close']] /= start_price

    processed['sma_200'] = processed['close'].rolling(window=200, min_periods=1).mean()
    processed['trend_ratio'] = processed['close'] / processed['sma_200']
    processed['trend_ratio'] = processed['trend_ratio'].fillna(1.0)
    
    # FINAL FEATURE LIST including 'rel_volume'
    my_tech_indicator_list = INDICATORS + ['vix', 'fear_signal', 'rate', 'rate_velocity', 'rel_volume', 'trend_ratio']
    
    # Data Prep
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data["date"].isin(processed["date"])]
    init_train_trade_data = init_train_trade_data.sort_values(["date", "tic"]).fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    
    env_kwargs = {
        "hmax": 500, 
        "initial_amount": 100000, 
        "num_stock_shares": [0]*stock_dimension,
        "buy_cost_pct": [0.0005]*stock_dimension, 
        "sell_cost_pct": [0.0005]*stock_dimension, 
        "state_space": state_space, 
        "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list, 
        "action_space": stock_dimension,
        "reward_scaling": 1e-3, 
    }

    # --- ENSEMBLE TRAINING ---
    ensemble_models = []
    seeds = [42, 101, 999] 
    
    if train_model:
        print(f"--- Training Macro + Volume Ensemble (3 Agents): {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        
        for seed in seeds:
            model = train_ensemble_agent(env_train, model_name, seed)
            ensemble_models.append(model)
            model.save(os.path.join(TRAINED_MODEL_DIR, f"{model_name}_seed{seed}"))
            
    else:
        print(f"Loading Ensemble: {model_name}")
        for seed in seeds:
            path = os.path.join(TRAINED_MODEL_DIR, f"{model_name}_seed{seed}")
            if os.path.exists(path + ".zip"):
                ensemble_models.append(PPO.load(path))
            else:
                print(f"Missing model: {path}")
                return 0.0

    # Prediction
    print(f"Backtesting with Ensemble Voting...")
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=None, risk_indicator_col=None, **env_kwargs)
    obs_trade, _ = e_trade_gym.reset()
    done = False
    
    while not done:
        actions_list = []
        for model in ensemble_models:
            action, _ = model.predict(obs_trade, deterministic=True)
            actions_list.append(action)
        avg_action = np.mean(actions_list, axis=0)
        
        obs_trade, rewards, dones, truncated, info = e_trade_gym.step(avg_action)
        done = dones if isinstance(dones, bool) else dones[0]

    # Logs
    df_actions = e_trade_gym.save_action_memory()
    df_account = e_trade_gym.save_asset_memory()
    
    df_actions = df_actions.reset_index(drop=True)
    df_account = df_account.reset_index(drop=True)
    df_actions.columns = TARGET_ASSETS 
    df_actions['date'] = df_account['date'].iloc[:len(df_actions)]
    
    significant_trades = df_actions[ (df_actions[TARGET_ASSETS].abs() > 10).any(axis=1) ]
    
    print("\n" + "="*40)
    print(f"      MACRO + VOLUME LOG ({trade_start_date[:4]})      ")
    print("="*40)
    if not significant_trades.empty:
        print(significant_trades.head(5).to_string(index=False))
        print("...")
        print(significant_trades.tail(5).to_string(index=False))
    else:
        print("No significant trades found.")
    print("="*40 + "\n")

    final_val = df_account['account_value'].iloc[-1]
    initial_val = df_account['account_value'].iloc[0]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    print(f"Run Finished. Return: {total_return_pct:.2f}%")
    return total_return_pct

if __name__ == "__main__":
    TEST_YEARS = [2019, 2020, 2021, 2022]
    overall_results = []
    
    print(f"Starting Global Macro + Volume Backtest...")
    
    for year in TEST_YEARS:
        print(f"\n=== TESTING YEAR: {year} ===")
        train_start, train_end = "2009-01-01", f"{year}-01-01"
        test_start, test_end = f"{year}-01-01", f"{year+1}-01-01"
        
        ai_return = stock_trading(
            train_start_date=train_start, train_end_date=train_end,
            trade_start_date=test_start, trade_end_date=test_end,
            train_model=True, model_name=f"ppo_macro_vol_{year}"
        )
        
        spy_loader = RobustYahooDownloader(start_date=test_start, end_date=test_end, ticker_list=["SPY"])
        spy_df = spy_loader.fetch_data()
        spy_return = ((spy_df.iloc[-1]['close'] - spy_df.iloc[0]['close']) / spy_df.iloc[0]['close']) * 100
        
        overall_results.append({
            "Year": year, "Model": f"{ai_return:.2f}%", "SPY": f"{spy_return:.2f}%",
            "Win": "YES" if ai_return > spy_return else "NO"
        })

    print("\n" + pd.DataFrame(overall_results).to_string(index=False))