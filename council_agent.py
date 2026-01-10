import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.config import DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR, INDICATORS
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# --- 1. CONFIGURATION ---
START_DATE = "2009-01-01" 
END_DATE = "2026-02-01"
UNIVERSE = ["XBI", "LABU", "LABD", "VRTX", "SHY"] 

# --- 2. CLASS DEFINITIONS ---
class RobustYahooDownloader:
    def __init__(self, start_date, end_date, ticker_list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self):
        print(f"Downloading {self.ticker_list} from {self.start_date} to {self.end_date}...")
        try:
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date, 
                ignore_tz=True, threads=False, progress=False, auto_adjust=True
            )
        except Exception as e:
            print(f"Download Error: {e}")
            return pd.DataFrame()
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()
            df = df.rename(columns={'Date': 'date', 'Index': 'date'})
            if 'tic' not in df.columns:
                df['tic'] = self.ticker_list[0]

        df.columns = df.columns.str.lower()
        df['date'] = df['date'].astype(str)
        required = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required if c in df.columns]
        df = df[final_cols]
        df = df.sort_values(['date', 'tic']).dropna().reset_index(drop=True)
        return df

def process_data(df):
    print("Processing indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=True, user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    processed = processed.fillna(0)
    
    # Custom Features
    processed['log_ret'] = np.log(processed['close'] / processed['close'].shift(1)).fillna(0)
    processed['volatility_21'] = processed.groupby('tic')['log_ret'].transform(lambda x: x.rolling(21).std())
    
    # Market Regime (Lagged)
    xbi = processed[processed['tic'] == 'XBI'].copy()
    xbi['sma_50'] = xbi['close'].rolling(50).mean()
    xbi['bull_market'] = np.where(xbi['close'] > xbi['sma_50'], 1.0, 0.0)
    xbi['bull_market'] = xbi['bull_market'].shift(1).fillna(0)
    
    processed = processed.merge(xbi[['date', 'bull_market']], on='date', how='left').fillna(0)
    processed.index = processed.date.factorize()[0]
    return processed

# --- 3. AGENT SETUP ---
def train_specialist(role, df, timesteps=15000):
    stock_dim = len(df.tic.unique())
    tech_list = INDICATORS + ['volatility_21', 'bull_market']
    state_space = 1 + 2*stock_dim + (len(tech_list)*stock_dim)
    
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 100000, 
        "num_stock_shares": [0] * stock_dim, 
        "buy_cost_pct": [0.0025]*stock_dim, 
        "sell_cost_pct": [0.0025]*stock_dim,
        "state_space": state_space, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    
    e_train = StockTradingEnv(df=df, **env_kwargs)
    env_train, _ = e_train.get_sb_env()
    
    # Updated Names Logic
    if role == "Trend_Agent": 
        # Formerly SURFER
        params = {"ent_coef": 0.005, "learning_rate": 0.0002, "n_steps": 2048}
    elif role == "Momentum_Agent": 
        # Formerly SNIPER
        params = {"ent_coef": 0.05, "learning_rate": 0.0005, "n_steps": 1024}
    else: 
        # Formerly SENTINEL (Conservative/Risk Agent)
        params = {"ent_coef": 0.01, "learning_rate": 0.00025, "n_steps": 2048}
    
    model = PPO("MlpPolicy", env_train, verbose=0, **params)
    model.learn(total_timesteps=timesteps)
    return model, env_kwargs

class SoftVotingManagerEnv(gym.Env):
    def __init__(self, df, agents, env_kwargs):
        super().__init__()
        self.df = df.copy()
        self.stock_env = StockTradingEnv(df=self.df, **env_kwargs)
        self.agents = agents
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.stock_env.observation_space
        
    def reset(self, seed=None, options=None):
        return self.stock_env.reset(seed=seed, options=options)
        
    def step(self, action):
        # Softmax normalization
        weights = np.exp(action - np.max(action)) / np.sum(np.exp(action - np.max(action)))
        
        idx = min(self.stock_env.day * self.stock_env.stock_dim, len(self.stock_env.df)-1)
        
        # If Bull Market, suppress the 3rd agent (Conservative_Agent)
        if self.stock_env.df.iloc[idx]['bull_market'] > 0.5:
            weights[2] *= 0.2 
            weights /= np.sum(weights)

        obs = self.stock_env.state
        obs_r = np.array(obs).reshape(1, -1)
        
        # Collect predictions from all agents
        # Note: The order depends on insertion order in 'run_15_year_backtest'
        acts = [agent.predict(obs_r, deterministic=True)[0] for agent in self.agents.values()]
        
        final_action = (acts[0]*weights[0]) + (acts[1]*weights[1]) + (acts[2]*weights[2])
        
        return self.stock_env.step(final_action[0])

# --- 4. THE LONG TRAINING LOOP (15 YEARS) ---
def run_15_year_backtest():
    check_and_make_directories([RESULTS_DIR])
    
    loader = RobustYahooDownloader(START_DATE, END_DATE, UNIVERSE)
    df_raw = loader.fetch_data()
    
    if df_raw.empty:
        print("ERROR: No data downloaded.")
        return

    df_proc = process_data(df_raw)
    
    unique_years = sorted(pd.to_datetime(df_proc['date']).dt.year.unique())
    print(f"Data Years Available: {unique_years}")
    
    train_window = 4
    results_log = []
    
    start_idx = 0
    while start_idx + train_window + 1 < len(unique_years):
        
        agent_years = unique_years[start_idx : start_idx + train_window]
        manager_year = unique_years[start_idx + train_window]
        test_year = unique_years[start_idx + train_window + 1]
        
        d_agents = df_proc[pd.to_datetime(df_proc['date']).dt.year.isin(agent_years)]
        d_manager = df_proc[pd.to_datetime(df_proc['date']).dt.year == manager_year]
        d_test = df_proc[pd.to_datetime(df_proc['date']).dt.year == test_year]

        # Safety Check for incomplete years (e.g., 2026)
        if len(d_test) < 10:
            print(f"\n[Info] Stopping backtest at {test_year}: Insufficient data (Current Incomplete Year).")
            break
        
        print(f"\n>>> CYCLE: Agents {agent_years} | Manager {manager_year} | TEST {test_year} <<<")

        for d in [d_agents, d_manager, d_test]:
            d.index = d.date.factorize()[0]
            
        # Initialize Agents with New Names
        # IMPORTANT: Order matters for the Voting Manager logic!
        agents = {}
        print("  Training Specialists...")
        
        # 1. Trend_Agent (formerly SURFER)
        agents["Trend_Agent"], env_conf = train_specialist("Trend_Agent", d_agents)
        
        # 2. Momentum_Agent (formerly SNIPER)
        agents["Momentum_Agent"], _ = train_specialist("Momentum_Agent", d_agents)
        
        # 3. Conservative_Agent (formerly SENTINEL) - This one gets suppressed in Bull markets
        agents["Conservative_Agent"], _ = train_specialist("Conservative_Agent", d_agents)
        
        print("  Training Manager...")
        mgr_env = SoftVotingManagerEnv(d_manager, agents, env_conf)
        mgr_model = PPO("MlpPolicy", DummyVecEnv([lambda: mgr_env]), verbose=0, learning_rate=0.0003)
        mgr_model.learn(total_timesteps=10000)
        
        print(f"  Executing {test_year} Test...")
        test_env = SoftVotingManagerEnv(d_test, agents, env_conf)
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = mgr_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = test_env.step(action)
            
        final_val = test_env.stock_env.asset_memory[-1]
        roi = (final_val - 100000) / 100000
        print(f"  --> {test_year} ROI: {roi*100:.2f}%")
        results_log.append({"Year": test_year, "ROI": roi})
        
        start_idx += 1 

    print("\n" + "="*40)
    print("15-YEAR WALK-FORWARD RESULTS")
    print("="*40)
    total_compound = 1.0
    for res in results_log:
        print(f"{res['Year']}: {res['ROI']*100:>6.2f}%")
        total_compound *= (1 + res['ROI'])
    
    print("-" * 40)
    print(f"Total Cumulative Return: {(total_compound - 1)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_15_year_backtest()