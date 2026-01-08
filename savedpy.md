*With fed with difference, less than market, 2.8%*

from __future__ import annotations

import itertools
import sys
import os

import pandas as pd
import yfinance as yf
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, INDICATORS, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        # Download using threads for speed
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=True
        )
        
        # FIX: Handle newer yfinance MultiIndex output
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        # Clean column names
        df.columns = df.columns.str.lower()
        
        # Rename 'adj close' if it exists
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        # Ensure date is string format
        df['date'] = df['date'].astype(str)
        
        # Filter for required columns only
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        # Sort and reset index
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    if_store_actions: bool = True, if_store_result: bool = True,
    if_using_a2c: bool = True, if_using_ddpg: bool = True,
    if_using_ppo: bool = True, if_using_sac: bool = True,
    if_using_td3: bool = True,
):
    # Ensure directories exist
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # SECTOR ETF LIST
    NEW_DOW_30_TICKER = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]

    # 1. Download Stock Data
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=NEW_DOW_30_TICKER
    ).fetch_data()
    
    # 2. Add Technical Indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # 3. Manually Download Macro Data
    print("Downloading Macro Data (VIX + Yield Curve)...")
    macro_tickers = ["^VIX", "^TNX", "^IRX"]
    
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    
    rename_map = {
        '^VIX': 'vix', 
        '^TNX': 'long_term_rate',
        '^IRX': 'fed_rate'
    }
    macro_df = macro_df.rename(columns=rename_map)
    
    # --- CRITICAL FIX: CALCULATE YIELD CURVE SPREAD ---
    # Instead of raw rates, we use the difference.
    # Yield Curve = 10 Year Rate - Fed Rate
    # If this is Negative, it usually predicts a recession.
    macro_df['yield_curve'] = macro_df['long_term_rate'] - macro_df['fed_rate']
    
    # We only need VIX and Yield Curve now
    macro_df = macro_df[['date', 'vix', 'yield_curve']]
    
    # Merge into main data
    processed = processed.merge(macro_df, on='date', how='left')
    
    # Fill missing values
    for col in ['vix', 'yield_curve']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()
    
    print("Macro data (VIX + Yield Curve) merged successfully.")

    # 4. Define the Features the AI should use
    # We use 'yield_curve' instead of the raw rates
    my_tech_indicator_list = INDICATORS + ['vix', 'yield_curve']

    # 5. Standard Data Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    if if_using_ppo:
        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        new_logger_ppo = configure(RESULTS_DIR + "/ppo", ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=50000)
        
        # Trading/Backtest
        e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
        result_ppo, actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
        if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
        
        if if_store_actions: actions_ppo.to_csv("actions_ppo.csv")
        result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
        
        # --- FIX: Manually download DJI Baseline ---
        print("Downloading Baseline (^DJI)...")
        dji_df = RobustYahooDownloader(
            start_date=trade_start_date, end_date=trade_end_date, ticker_list=["^DJI"]
        ).fetch_data()
        
        dji = dji_df[['date', 'close']].rename(columns={'close': 'DJI'})
        # -------------------------------------------
        
        result = pd.merge(dji, result_ppo, on='date', how='left').dropna()
        result["DJI"] = result["DJI"] / result["DJI"].iloc[0] * initial_amount
        
        print("Result Head:\n", result.head())
        if if_store_result: result.to_csv("result.csv")
        
        try:
            plot_return(result=result, column_as_x=date_col, if_need_calc_return=True, savefig_filename="stock_trading.png", xlabel="Date", ylabel="Return")
        except Exception as e:
            print(f"Plotting failed (likely due to internal download), but results are saved to result.csv. Error: {e}")
if __name__ == "__main__":
    stock_trading(
        train_start_date="2009-01-01", train_end_date="2022-09-01",
        trade_start_date="2022-09-01", trade_end_date="2023-11-01",
        if_using_ppo=True, if_using_a2c=False, if_using_ddpg=False, 
        if_using_sac=False, if_using_td3=False
    )


*Current best, ~7% beats market*
from __future__ import annotations

import itertools
import sys
import os

import pandas as pd
import yfinance as yf
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, INDICATORS, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        # Download using threads for speed
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=True
        )
        
        # FIX: Handle newer yfinance MultiIndex output
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        # Clean column names
        df.columns = df.columns.str.lower()
        
        # Rename 'adj close' if it exists
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        # Ensure date is string format
        df['date'] = df['date'].astype(str)
        
        # Filter for required columns only
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        # Sort and reset index
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    if_store_actions: bool = True, if_store_result: bool = True,
    if_using_a2c: bool = True, if_using_ddpg: bool = True,
    if_using_ppo: bool = True, if_using_sac: bool = True,
    if_using_td3: bool = True,
):
    # Ensure directories exist
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # MANUAL LIST: Removed 'WBA'
   # SECTOR ETF LIST (11 Major US Sectors)
    # This replaces single stocks with entire economic sectors to avoid bankruptcy risk.
    NEW_DOW_30_TICKER = [
        "XLK",  # Technology (Apple, Microsoft, Nvidia)
        "XLV",  # Healthcare (Johnson & Johnson, Pfizer)
        "XLF",  # Financials (JPMorgan, Berkshire Hathaway)
        "XLE",  # Energy (Exxon, Chevron)
        "XLC",  # Communication Services (Google, Meta, Disney)
        "XLY",  # Consumer Discretionary (Amazon, Tesla, Home Depot)
        "XLP",  # Consumer Staples (P&G, Coca-Cola, Walmart)
        "XLI",  # Industrials (Boeing, Caterpillar)
        "XLB",  # Materials (Chemicals, Mining)
        "XLRE", # Real Estate (REITs)
        "XLU"   # Utilities (Electric, Gas)
    ]

    # 1. Download Stock Data
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=NEW_DOW_30_TICKER
    ).fetch_data()
    
    # 2. Add Technical Indicators (use_vix=False to prevent crash)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # 3. Manually Download Macro Data (VIX + Interest Rates)
    # ^VIX = Volatility Index (Fear Gauge)
    # ^TNX = 10-Year Treasury Yield (Interest Rates)
    print("Downloading Macro Data (VIX + 10-Year Treasury Yield)...")
    macro_tickers = ["^VIX", "^TNX"]
    
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    # Pivot macro data so we have simple columns [date, ^TNX, ^VIX]
    # We use 'close' price for these indicators
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    
    # Rename columns to be friendly features
    # Note: Depending on download order, columns might be different, so we map explicitly
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate'}
    macro_df = macro_df.rename(columns=rename_map)
    
    # Merge these new features into the main stock data
    # (Left join ensures we keep the stock data structure)
    processed = processed.merge(macro_df, on='date', how='left')
    
    # Fill missing values (forward fill then backward fill)
    if 'vix' in processed.columns:
        processed['vix'] = processed['vix'].ffill().bfill()
    if 'interest_rate' in processed.columns:
        processed['interest_rate'] = processed['interest_rate'].ffill().bfill()
    
    print("Macro data merged successfully.")

    # 4. Define the Features the AI should actually use
    # We take the standard indicators (MACD, RSI, etc) AND add our two new ones
    my_tech_indicator_list = INDICATORS + ['vix', 'interest_rate']

    # 5. Standard Data Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    # UPDATE STATE SPACE: We must calculate using our NEW list of indicators
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list, # <--- Pass the UPDATED list here
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    if if_using_ppo:
        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        new_logger_ppo = configure(RESULTS_DIR + "/ppo", ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=50000)
        
        # Trading/Backtest
        e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
        result_ppo, actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
        if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
        
        if if_store_actions: actions_ppo.to_csv("actions_ppo.csv")
        result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
        
        # --- FIX: Manually download DJI Baseline ---
        print("Downloading Baseline (^DJI)...")
        dji_df = RobustYahooDownloader(
            start_date=trade_start_date, end_date=trade_end_date, ticker_list=["^DJI"]
        ).fetch_data()
        
        dji = dji_df[['date', 'close']].rename(columns={'close': 'DJI'})
        # -------------------------------------------
        
        result = pd.merge(dji, result_ppo, on='date', how='left').dropna()
        result["DJI"] = result["DJI"] / result["DJI"].iloc[0] * initial_amount
        
        print("Result Head:\n", result.head())
        if if_store_result: result.to_csv("result.csv")
        
        try:
            plot_return(result=result, column_as_x=date_col, if_need_calc_return=True, savefig_filename="stock_trading.png", xlabel="Date", ylabel="Return")
        except Exception as e:
            print(f"Plotting failed (likely due to internal download), but results are saved to result.csv. Error: {e}")

if __name__ == "__main__":
    stock_trading(
        train_start_date="2009-01-01", train_end_date="2022-09-01",
        trade_start_date="2022-09-01", trade_end_date="2023-11-01",
        if_using_ppo=True, if_using_a2c=False, if_using_ddpg=False, 
        if_using_sac=False, if_using_td3=False
    )

*With ETF comparsions(health drop vs finance increase?)*
from __future__ import annotations

import itertools
import sys
import os

import pandas as pd
import yfinance as yf
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, INDICATORS, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=True
        )
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    if_store_actions: bool = True, if_store_result: bool = True,
    if_using_a2c: bool = True, if_using_ddpg: bool = True,
    if_using_ppo: bool = True, if_using_sac: bool = True,
    if_using_td3: bool = True,
):
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # SECTOR ETF LIST
    NEW_DOW_30_TICKER = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]

    # 1. Download Stock Data
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=NEW_DOW_30_TICKER
    ).fetch_data()
    
    # 2. Add Technical Indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # ---------------------------------------------------------
    # STEP 2.5: ADD SECTOR LEAD-LAG FEATURES
    # We calculate the "Tech Sector (XLK)" momentum from 30 days ago
    # and add it to EVERY row. This lets the AI see if Tech predicts other sectors.
    # ---------------------------------------------------------
    print("Calculating Lead-Lag Features (Tech Sector Momentum)...")
    
    # Isolate just the Tech Sector (XLK)
    tech_df = processed[processed['tic'] == 'XLK'].copy()
    tech_df = tech_df[['date', 'close']].rename(columns={'close': 'xlk_close'})
    
    # Calculate 30-day return for Tech (The "Leading" Indicator)
    # pct_change(30) means "Return over the last 30 days"
    # shift(1) ensures we only know yesterday's data (no cheating!)
    tech_df['tech_momentum_30d'] = tech_df['xlk_close'].pct_change(30).shift(1)
    
    # Only keep the date and the new feature
    tech_feature = tech_df[['date', 'tech_momentum_30d']]
    
    # Merge this "Global Tech Signal" into the main dataset
    # Now EVERY sector (Energy, Utilities, etc.) knows how Tech did last month.
    processed = processed.merge(tech_feature, on='date', how='left')
    
    # Fill NaN values (first 30 days will be empty)
    processed['tech_momentum_30d'] = processed['tech_momentum_30d'].fillna(0)
    print("Lead-Lag features added.")
    # ---------------------------------------------------------

    # 3. Add Simple Macro Data (VIX + TNX) - keeping the "Goldilocks" macro data
    print("Downloading Simple Macro Data...")
    macro_tickers = ["^VIX", "^TNX"]
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate'}
    macro_df = macro_df.rename(columns=rename_map)
    processed = processed.merge(macro_df, on='date', how='left')
    
    for col in ['vix', 'interest_rate']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()

    # 4. Define Features (Standard + Macro + New Lead/Lag)
    my_tech_indicator_list = INDICATORS + ['vix', 'interest_rate', 'tech_momentum_30d']

    # 5. Standard Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    if if_using_ppo:
        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        new_logger_ppo = configure(RESULTS_DIR + "/ppo", ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=50000)
        
        e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
        result_ppo, actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
        if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
        
        if if_store_actions: actions_ppo.to_csv("actions_ppo.csv")
        result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
        
        print("Downloading Baseline (^DJI)...")
        dji_df = RobustYahooDownloader(
            start_date=trade_start_date, end_date=trade_end_date, ticker_list=["^DJI"]
        ).fetch_data()
        dji = dji_df[['date', 'close']].rename(columns={'close': 'DJI'})
        
        result = pd.merge(dji, result_ppo, on='date', how='left').dropna()
        result["DJI"] = result["DJI"] / result["DJI"].iloc[0] * initial_amount
        
        print("Result Head:\n", result.head())
        if if_store_result: result.to_csv("result.csv")
        
        try:
            plot_return(result=result, column_as_x=date_col, if_need_calc_return=True, savefig_filename="stock_trading.png", xlabel="Date", ylabel="Return")
        except Exception as e:
            print(f"Plotting failed (likely due to internal download), but results are saved to result.csv. Error: {e}")

if __name__ == "__main__":
    stock_trading(
        train_start_date="2009-01-01", train_end_date="2022-09-01",
        trade_start_date="2022-09-01", trade_end_date="2023-11-01",
        if_using_ppo=True, if_using_a2c=False, if_using_ddpg=False, 
        if_using_sac=False, if_using_td3=False
    )


*with testing* 1.3%
from __future__ import annotations

import itertools
import sys
import os
import numpy as np # Needed for calculating averages

import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, INDICATORS, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        # threads=False is slower but prevents 'Operation timed out' errors
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=False 
        )
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        # Drop Duplicates and Missing Data to prevent crashes
        df = df.drop_duplicates(subset=['date', 'tic'])
        df = df.dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float: # Returns the final profit percentage
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # SECTOR ETF LIST
    NEW_DOW_30_TICKER = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]

    # 1. Download Stock Data
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=NEW_DOW_30_TICKER
    ).fetch_data()
    
    # 2. Add Technical Indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # 3. Add Lead-Lag Features (Tech Momentum)
    print("Calculating Lead-Lag Features...")
    tech_df = processed[processed['tic'] == 'XLK'].copy()
    tech_df = tech_df[['date', 'close']].rename(columns={'close': 'xlk_close'})
    tech_df['tech_momentum_30d'] = tech_df['xlk_close'].pct_change(30).shift(1)
    tech_feature = tech_df[['date', 'tech_momentum_30d']]
    processed = processed.merge(tech_feature, on='date', how='left')
    processed['tech_momentum_30d'] = processed['tech_momentum_30d'].fillna(0)

    # 4. Add Macro Data (VIX + TNX)
    print("Downloading Macro Data...")
    macro_tickers = ["^VIX", "^TNX"]
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate'}
    macro_df = macro_df.rename(columns=rename_map)
    processed = processed.merge(macro_df, on='date', how='left')
    
    for col in ['vix', 'interest_rate']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()

    # 5. Define Features
    my_tech_indicator_list = INDICATORS + ['vix', 'interest_rate', 'tech_momentum_30d']

    # 6. Standard Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # --- TRAIN OR LOAD LOGIC ---
    if train_model:
        print(f"--- Training Model: {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        # Suppress logging for speed in loop
        new_logger_ppo = configure(os.path.join(RESULTS_DIR, model_name), ["stdout", "csv"])
        model_ppo.set_logger(new_logger_ppo)
        
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name=model_name, total_timesteps=50000)
        
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        trained_ppo.save(save_path)
        print(f"Saved to: {save_path}")
        
    else:
        print(f"Loading model: {model_name}")
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if os.path.exists(save_path + ".zip"):
            trained_ppo = PPO.load(save_path)
        else:
            print(f"Error: {save_path} not found.")
            return 0.0

    # --- BACKTESTING ---
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
    result_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
    
    result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
    
    # Calculate Final Return %
    final_val = result_ppo['PPO'].iloc[-1]
    initial_val = result_ppo['PPO'].iloc[0]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    print(f"Run Finished. Return: {total_return_pct:.2f}%")
    return total_return_pct

if __name__ == "__main__":
    # --- EXPERIMENT CONFIGURATION ---
    NUM_RUNS = 5  # How many times to train?
    TRAIN_MODE = True # Set to False to load previous 5 models instead
    
    results = []
    
    print(f"Starting experiment with {NUM_RUNS} runs...")
    
    for i in range(NUM_RUNS):
        unique_model_name = f"sector_rotation_run_{i}"
        
        # Run the training/testing
        run_return = stock_trading(
            train_start_date="2009-01-01", train_end_date="2022-09-01",
            trade_start_date="2022-09-01", trade_end_date="2023-11-01",
            train_model=TRAIN_MODE,
            model_name=unique_model_name
        )
        results.append(run_return)
        print("-" * 30)

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("      EXPERIMENT RESULTS      ")
    print("="*30)
    print(f"Individual Returns: {results}")
    print(f"Average Return:     {np.mean(results):.2f}%")
    print(f"Std Deviation:      {np.std(results):.2f}%")
    print(f"Best Run:           {np.max(results):.2f}%")
    print(f"Worst Run:          {np.min(results):.2f}%")
    print("="*30)


**delta with avg, bad* 0.28%**
from __future__ import annotations

import itertools
import sys
import os
import numpy as np # Needed for calculating averages

import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, INDICATORS, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        # threads=False is slower but prevents 'Operation timed out' errors
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=False 
        )
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        # Drop Duplicates and Missing Data to prevent crashes
        df = df.drop_duplicates(subset=['date', 'tic'])
        df = df.dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float: # Returns the final profit percentage
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # SECTOR ETF LIST
    NEW_DOW_30_TICKER = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]

    # 1. Download Stock Data
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=NEW_DOW_30_TICKER
    ).fetch_data()
    
    # 2. Add Technical Indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

   # 3. (SKIPPED) We don't use Lead-Lag features for this version.

    # 4. Add Macro Data (VIX + TNX + Fed Rate Change)
    print("Downloading Macro Data (VIX + TNX + Fed Rate)...")
    macro_tickers = ["^VIX", "^TNX", "^IRX"]
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate', '^IRX': 'fed_rate_raw'}
    macro_df = macro_df.rename(columns=rename_map)
    
    # CALCULATE DELTA (Daily Change)
    # This turns "5.0%" into "+0.25" so the AI doesn't panic at high rates
    macro_df['fed_rate_delta'] = macro_df['fed_rate_raw'].diff().fillna(0)
    
    # Drop the raw rate so the AI never sees the scary big number
    macro_df = macro_df[['date', 'vix', 'interest_rate', 'fed_rate_delta']]
    
    processed = processed.merge(macro_df, on='date', how='left')
    
    for col in ['vix', 'interest_rate', 'fed_rate_delta']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()

    # 5. Define Features (Use the Delta!)
    my_tech_indicator_list = INDICATORS + ['vix', 'interest_rate', 'fed_rate_delta']

    # 6. Standard Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # --- TRAIN OR LOAD LOGIC ---
    if train_model:
        print(f"--- Training Model: {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        # Suppress logging for speed in loop
        new_logger_ppo = configure(os.path.join(RESULTS_DIR, model_name), ["stdout", "csv"])
        model_ppo.set_logger(new_logger_ppo)
        
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name=model_name, total_timesteps=50000)
        
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        trained_ppo.save(save_path)
        print(f"Saved to: {save_path}")
        
    else:
        print(f"Loading model: {model_name}")
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if os.path.exists(save_path + ".zip"):
            trained_ppo = PPO.load(save_path)
        else:
            print(f"Error: {save_path} not found.")
            return 0.0

    # --- BACKTESTING ---
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
    result_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
    
    result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
    
    # Calculate Final Return %
    final_val = result_ppo['PPO'].iloc[-1]
    initial_val = result_ppo['PPO'].iloc[0]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    print(f"Run Finished. Return: {total_return_pct:.2f}%")
    return total_return_pct

if __name__ == "__main__":
    # --- EXPERIMENT CONFIGURATION ---
    NUM_RUNS = 5  # How many times to train?
    TRAIN_MODE = True # Set to False to load previous 5 models instead
    
    results = []
    
    print(f"Starting experiment with {NUM_RUNS} runs...")
    
    for i in range(NUM_RUNS):
        unique_model_name = f"sector_rotation_run_{i}"
        
        # Run the training/testing
        run_return = stock_trading(
            train_start_date="2009-01-01", train_end_date="2022-09-01",
            trade_start_date="2022-09-01", trade_end_date="2023-11-01",
            train_model=TRAIN_MODE,
            model_name=unique_model_name
        )
        results.append(run_return)
        print("-" * 30)

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("      EXPERIMENT RESULTS      ")
    print("="*30)
    print(f"Individual Returns: {results}")
    print(f"Average Return:     {np.mean(results):.2f}%")
    print(f"Std Deviation:      {np.std(results):.2f}%")
    print(f"Best Run:           {np.max(results):.2f}%")
    print(f"Worst Run:          {np.min(results):.2f}%")
    print("="*30)

**very good, around ~9% using sector etfs to predict spy500*
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
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=False # Safer
        )
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        df = df.drop_duplicates(subset=['date', 'tic'])
        df = df.dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float:
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # 1. Download TARGET Data (SPY only)
    # The AI will only actually BUY and SELL this one stock.
    print("Downloading Target (SPY)...")
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=["SPY"]
    ).fetch_data()
    
    # 2. Add Technical Indicators to SPY
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # 3. THE STRATEGY: Download Sectors as FEATURES (Inputs)
    # We download the 11 sectors, but we don't trade them. 
    # We use them to predict SPY.
    print("Downloading Sector Factors (Inputs)...")
    SECTOR_TICKERS = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]
    
    sector_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=SECTOR_TICKERS
    ).fetch_data()
    
    # Pivot so each sector is a column
    # We use 'close' price
    sector_wide = sector_df.pivot(index='date', columns='tic', values='close')
    
    # CRITICAL STEP: Calculate Daily Returns (%)
    # The AI doesn't care that Tech is $150 and Energy is $80.
    # It cares that Tech is UP 1% and Energy is DOWN 2%.
    sector_returns = sector_wide.pct_change()
    
    # Rename columns to be clear (e.g., 'XLK' -> 'return_XLK')
    sector_returns.columns = [f'return_{col}' for col in sector_returns.columns]
    sector_returns = sector_returns.reset_index()
    
    # Merge these 11 new "Features" into the SPY data
    processed = processed.merge(sector_returns, on='date', how='left')
    processed = processed.fillna(0) # Fill the first day (NaN) with 0
    
    print("Sector factors merged.")

    # 4. Add Macro Data (Just VIX + TNX) - The "Goldilocks" Macro
    print("Downloading Macro Data...")
    macro_tickers = ["^VIX", "^TNX"]
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate'}
    macro_df = macro_df.rename(columns=rename_map)
    processed = processed.merge(macro_df, on='date', how='left')
    
    for col in ['vix', 'interest_rate']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()

    # 5. Define The Massive Feature List
    # Standard Indicators + 11 Sector Returns + VIX + Interest Rate
    sector_features = [f'return_{tic}' for tic in SECTOR_TICKERS]
    my_tech_indicator_list = INDICATORS + sector_features + ['vix', 'interest_rate']
    
    print(f"Total Features per Step: {len(my_tech_indicator_list)}")

    # 6. Standard Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # --- TRAIN OR LOAD LOGIC ---
    if train_model:
        print(f"--- Training Model: {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        new_logger_ppo = configure(os.path.join(RESULTS_DIR, model_name), ["stdout", "csv"])
        model_ppo.set_logger(new_logger_ppo)
        
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name=model_name, total_timesteps=50000)
        
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        trained_ppo.save(save_path)
        print(f"Saved to: {save_path}")
        
    else:
        print(f"Loading model: {model_name}")
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if os.path.exists(save_path + ".zip"):
            trained_ppo = PPO.load(save_path)
        else:
            print(f"Error: {save_path} not found.")
            return 0.0

    # --- BACKTESTING ---
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
    result_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
    
    result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
    
    final_val = result_ppo['PPO'].iloc[-1]
    initial_val = result_ppo['PPO'].iloc[0]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    print(f"Run Finished. Return: {total_return_pct:.2f}%")
    return total_return_pct

if __name__ == "__main__":
    # --- EXPERIMENT CONFIGURATION ---
    NUM_RUNS = 5  
    TRAIN_MODE = True 
    
    results = []
    
    print(f"Starting experiment with {NUM_RUNS} runs...")
    
    for i in range(NUM_RUNS):
        unique_model_name = f"spy_factor_model_run__try5{i}"
        
        run_return = stock_trading(
            train_start_date="2009-01-01", train_end_date="2022-09-01",
            trade_start_date="2022-09-01", trade_end_date="2023-11-01",
            train_model=TRAIN_MODE,
            model_name=unique_model_name
        )
        results.append(run_return)
        print("-" * 30)

    print("\n" + "="*30)
    print("      EXPERIMENT RESULTS      ")
    print("="*30)
    print(f"Individual Returns: {results}")
    print(f"Average Return:     {np.mean(results):.2f}%")
    print(f"Std Deviation:      {np.std(results):.2f}%")
    print("="*30)

**with fed, i think it went down a little*
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
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=False # Safer
        )
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        df = df.drop_duplicates(subset=['date', 'tic'])
        df = df.dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float:
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # 1. Download TARGET Data (SPY only)
    print("Downloading Target (SPY)...")
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=["SPY"]
    ).fetch_data()
    
    # 2. Add Technical Indicators to SPY
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # 3. THE STRATEGY: Download Sectors as FEATURES
    print("Downloading Sector Factors (Inputs)...")
    SECTOR_TICKERS = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]
    
    sector_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=SECTOR_TICKERS
    ).fetch_data()
    
    # Pivot and Calculate Daily Returns
    sector_wide = sector_df.pivot(index='date', columns='tic', values='close')
    sector_returns = sector_wide.pct_change()
    
    # Rename columns (e.g., 'XLK' -> 'return_XLK')
    sector_returns.columns = [f'return_{col}' for col in sector_returns.columns]
    sector_returns = sector_returns.reset_index()
    
    # Merge Features
    processed = processed.merge(sector_returns, on='date', how='left')
    processed = processed.fillna(0)
    
    print("Sector factors merged.")

    # 4. Add Macro Data (VIX + TNX + FED RATE DELTA)
    print("Downloading Macro Data (VIX + Interest Rates)...")
    macro_tickers = ["^VIX", "^TNX", "^IRX"] # Added ^IRX back
    
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate', '^IRX': 'fed_rate_raw'}
    macro_df = macro_df.rename(columns=rename_map)
    
    # --- CALCULATE FED RATE DELTA ---
    # We use the change (diff) so the AI isn't scared of the 5% absolute number
    macro_df['fed_rate_delta'] = macro_df['fed_rate_raw'].diff().fillna(0)
    
    # Keep only the columns we need
    macro_df = macro_df[['date', 'vix', 'interest_rate', 'fed_rate_delta']]
    
    processed = processed.merge(macro_df, on='date', how='left')
    
    # Fill missing values
    for col in ['vix', 'interest_rate', 'fed_rate_delta']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()

    # 5. Define The Feature List
    # Standard + Sectors + VIX + Interest Rate + Fed Rate Change
    sector_features = [f'return_{tic}' for tic in SECTOR_TICKERS]
    my_tech_indicator_list = INDICATORS + sector_features + ['vix', 'interest_rate', 'fed_rate_delta']
    
    print(f"Total Features per Step: {len(my_tech_indicator_list)}")

    # 6. Standard Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # --- TRAIN OR LOAD LOGIC ---
    if train_model:
        print(f"--- Training Model: {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        new_logger_ppo = configure(os.path.join(RESULTS_DIR, model_name), ["stdout", "csv"])
        model_ppo.set_logger(new_logger_ppo)
        
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name=model_name, total_timesteps=50000)
        
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        trained_ppo.save(save_path)
        print(f"Saved to: {save_path}")
        
    else:
        print(f"Loading model: {model_name}")
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if os.path.exists(save_path + ".zip"):
            trained_ppo = PPO.load(save_path)
        else:
            print(f"Error: {save_path} not found.")
            return 0.0

    # --- BACKTESTING ---
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
    result_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
    
    result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
    
    final_val = result_ppo['PPO'].iloc[-1]
    initial_val = result_ppo['PPO'].iloc[0]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    print(f"Run Finished. Return: {total_return_pct:.2f}%")
    return total_return_pct

if __name__ == "__main__":
    # --- EXPERIMENT CONFIGURATION ---
    NUM_RUNS = 5 
    TRAIN_MODE = True 
    
    results = []
    
    print(f"Starting experiment with {NUM_RUNS} runs...")
    
    for i in range(NUM_RUNS):
        unique_model_name = f"spy_fed_model_run_big{i}"
        
        run_return = stock_trading(
            train_start_date="2009-01-01", train_end_date="2022-09-01",
            trade_start_date="2022-09-01", trade_end_date="2023-11-01",
            train_model=TRAIN_MODE,
            model_name=unique_model_name
        )
        results.append(run_return)
        print("-" * 30)

    print("\n" + "="*30)
    print("      EXPERIMENT RESULTS      ")
    print("="*30)
    print(f"Individual Returns: {results}")
    print(f"Average Return:     {np.mean(results):.2f}%")
    print(f"Std Deviation:      {np.std(results):.2f}%")
    print("="*30)

**decent**
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
from finrl.plot import backtest_stats, plot_return

# --- CUSTOM CLASS TO FIX YAHOO CRASHES ---
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        raw_df = yf.download(
            self.ticker_list, 
            start=self.start_date, 
            end=self.end_date,
            ignore_tz=True,
            threads=False 
        )
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            df = raw_df.stack(level=1, future_stack=True)
            df.index.names = ['date', 'tic']
            df = df.reset_index()
        else:
            df = raw_df.reset_index()

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        df['date'] = df['date'].astype(str)
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        df = df[final_cols]
        
        df = df.drop_duplicates(subset=['date', 'tic'])
        df = df.dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Download complete. Data shape: {df.shape}")
        return df
# ----------------------------------------------

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float:
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    date_col = "date"
    tic_col = "tic"
    
    # 1. Download Target (SPY)
    print("Downloading Target (SPY)...")
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=["SPY"]
    ).fetch_data()
    
    # 2. Add Technical Indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # Trend Detector
    processed['sma_200'] = processed['close'].rolling(window=200, min_periods=1).mean()
    processed['trend_ratio'] = processed['close'] / processed['sma_200']
    processed['trend_ratio'] = processed['trend_ratio'].fillna(1.0)
    
    # 3. Download Sector Factors
    print("Downloading Sector Factors...")
    SECTOR_TICKERS = [
        "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", 
        "XLP", "XLI", "XLB", "XLRE", "XLU"
    ]
    
    sector_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=SECTOR_TICKERS
    ).fetch_data()
    
    sector_wide = sector_df.pivot(index='date', columns='tic', values='close')
    sector_returns = sector_wide.pct_change()
    sector_returns.columns = [f'return_{col}' for col in sector_returns.columns]
    sector_returns = sector_returns.reset_index()
    
    processed = processed.merge(sector_returns, on='date', how='left')
    processed = processed.fillna(0)
    
    # 4. Add Macro Data
    print("Downloading Macro Data...")
    macro_tickers = ["^VIX", "^TNX", "^IRX"]
    macro_df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=macro_tickers
    ).fetch_data()
    
    macro_df = macro_df.pivot(index='date', columns='tic', values='close').reset_index()
    rename_map = {'^VIX': 'vix', '^TNX': 'interest_rate', '^IRX': 'fed_rate_raw'}
    macro_df = macro_df.rename(columns=rename_map)
    
    macro_df['fed_rate_delta'] = macro_df['fed_rate_raw'].diff().fillna(0)
    macro_df = macro_df[['date', 'vix', 'interest_rate', 'fed_rate_delta']]
    
    processed = processed.merge(macro_df, on='date', how='left')
    
    for col in ['vix', 'interest_rate', 'fed_rate_delta']:
        if col in processed.columns:
            processed[col] = processed[col].ffill().bfill()

    # 5. Define Feature List
    sector_features = [f'return_{tic}' for tic in SECTOR_TICKERS]
    my_tech_indicator_list = INDICATORS + sector_features + ['vix', 'interest_rate', 'fed_rate_delta', 'trend_ratio']
    
    print(f"Total Features per Step: {len(my_tech_indicator_list)}")

    # 6. Processing
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=[date_col, tic_col]).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[init_train_trade_data[date_col].isin(processed[date_col])]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])
    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(init_train_trade_data, train_start_date, train_end_date)
    init_trade_data = data_split(init_train_trade_data, trade_start_date, trade_end_date)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(my_tech_indicator_list) * stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    initial_amount = 1000000
    
    env_kwargs = {
        "hmax": 100, "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # --- TRAIN OR LOAD LOGIC ---
    if train_model:
        print(f"--- Training Model (PPO Investor): {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        
        # --- PPO PARAMS (Champion Settings) ---
        PPO_PARAMS = {
            "n_steps": 2048, 
            "ent_coef": 0.001, # Lowered from 0.005 to reduce randomness
            "learning_rate": 0.0001, 
            "batch_size": 128,
            "gamma": 0.995 
        }
        
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        new_logger = configure(os.path.join(RESULTS_DIR, model_name), ["stdout", "csv"])
        model_ppo.set_logger(new_logger)
        
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name=model_name, total_timesteps=50000)
        
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        trained_ppo.save(save_path)
        print(f"Saved to: {save_path}")
        
    else:
        print(f"Loading model: {model_name}")
        save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if os.path.exists(save_path + ".zip"):
            trained_ppo = PPO.load(save_path)
        else:
            print(f"Error: {save_path} not found.")
            return 0.0

    # --- BACKTESTING ---
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=150, risk_indicator_col="vix", **env_kwargs)
    result_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if isinstance(result_ppo, tuple): result_ppo = result_ppo[0]
    
    result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
    
    final_val = result_ppo['PPO'].iloc[-1]
    initial_val = result_ppo['PPO'].iloc[0]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100
    
    print(f"Run Finished. Return: {total_return_pct:.2f}%")
    return total_return_pct

if __name__ == "__main__":
    TEST_YEARS = [2019, 2020, 2021, 2022]
    
    overall_results = []
    
    print(f"Starting Rolling Window Backtest (PPO Champion) on {len(TEST_YEARS)} years...")
    
    for year in TEST_YEARS:
        print(f"\n" + "="*40)
        print(f"      TESTING YEAR: {year}")
        print("="*40)
        
        train_start = "2009-01-01"
        train_end   = f"{year}-01-01"
        test_start  = f"{year}-01-01"
        test_end    = f"{year+1}-01-01"
        
        model_filename = f"ppo_investor_{year}"
        
        ai_return = stock_trading(
            train_start_date=train_start, train_end_date=train_end,
            trade_start_date=test_start, trade_end_date=test_end,
            train_model=True, 
            model_name=model_filename
        )
        
        print(f"Calculating Benchmark (SPY) for {year}...")
        spy_loader = RobustYahooDownloader(
            start_date=test_start, end_date=test_end, ticker_list=["SPY"]
        )
        spy_df = spy_loader.fetch_data()
        
        spy_open = spy_df.iloc[0]['close']
        spy_close = spy_df.iloc[-1]['close']
        spy_return = ((spy_close - spy_open) / spy_open) * 100
        
        overall_results.append({
            "Year": year,
            "Model Return": f"{ai_return:.2f}%",
            "SPY Return": f"{spy_return:.2f}%",
            "Beat Market?": "YES" if ai_return > spy_return else "NO"
        })

    print("\n" + "="*50)
    print("      ROLLING WINDOW REPORT CARD (PPO)      ")
    print("="*50)
    
    df_results = pd.DataFrame(overall_results)
    print(df_results.to_string(index=False))
    
    wins = df_results[df_results["Beat Market?"] == "YES"]
    print("-" * 50)
    print(f"Final Score: You beat the market in {len(wins)}/{len(TEST_YEARS)} years.")
    print("="*50)


**saved decent similar to above**
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
        raw_df = yf.download(
            self.ticker_list, start=self.start_date, end=self.end_date,
            ignore_tz=True, threads=False 
        )
        
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

def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float:
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # --- STRATEGIC UPDATE: MOMENTUM ONLY ---
    # SPY = S&P 500 (Baseline)
    # QQQ = Nasdaq 100 (Turbo Boost)
    # No Cash. No Bonds. Only Growth.
    TARGET_ASSETS = ["SPY", "QQQ"]
    
    print(f"Fetching Data for {TARGET_ASSETS}...")
    df = RobustYahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=TARGET_ASSETS
    ).fetch_data()
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=False, user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # 2. SMART SCALING
    print("Applying Smart Scaling...")
    for tic in processed['tic'].unique():
        mask = processed['tic'] == tic
        if mask.any():
            start_price = processed.loc[mask, 'close'].iloc[0]
            processed.loc[mask, ['open', 'high', 'low', 'close']] /= start_price
            
    # Trend Context
    processed['sma_200'] = processed['close'].rolling(window=200, min_periods=1).mean()
    processed['trend_ratio'] = processed['close'] / processed['sma_200']
    processed['trend_ratio'] = processed['trend_ratio'].fillna(1.0)
    
    my_tech_indicator_list = INDICATORS + ['trend_ratio']
    
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
        "buy_cost_pct": [0.0]*stock_dimension, # FREE TRADING to encourage switching
        "sell_cost_pct": [0.0]*stock_dimension, 
        "state_space": state_space, 
        "stock_dim": stock_dimension,
        "tech_indicator_list": my_tech_indicator_list, 
        "action_space": stock_dimension,
        "reward_scaling": 1e-3, 
    }

    save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
    
    # 4. TRAINING
    if train_model:
        print(f"--- Training Model (PPO): {model_name} ---")
        e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        
        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {
            "n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, 
            "batch_size": 64, "gamma": 0.999 
        }
        
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        new_logger = configure(os.path.join(RESULTS_DIR, model_name), ["stdout", "csv"])
        model_ppo.set_logger(new_logger)
        
        trained_ppo = agent.train_model(model=model_ppo, tb_log_name=model_name, total_timesteps=80000)
        trained_ppo.save(save_path)
    else:
        print(f"Loading model: {model_name}")
        if not os.path.exists(save_path + ".zip"):
            return 0.0
        trained_ppo = PPO.load(save_path)

    # 5. BACKTESTING WITH FIXED LOGS
    print(f"Backtesting on {trade_start_date} to {trade_end_date}...")

    e_trade_gym = StockTradingEnv(
        df=init_trade_data, 
        turbulence_threshold=None, 
        risk_indicator_col=None, 
        **env_kwargs
    )
    
    obs_trade, _ = e_trade_gym.reset()
    done = False
    
    while not done:
        action, _states = trained_ppo.predict(obs_trade, deterministic=True)
        obs_trade, rewards, dones, truncated, info = e_trade_gym.step(action)
        done = dones if isinstance(dones, bool) else dones[0]

    # --- FIX FOR LOGS ---
    df_actions = e_trade_gym.save_action_memory()
    df_account = e_trade_gym.save_asset_memory()
    
    df_actions = df_actions.reset_index(drop=True)
    df_account = df_account.reset_index(drop=True)
    df_actions.columns = TARGET_ASSETS 
    df_actions['date'] = df_account['date'].iloc[:len(df_actions)]
    
    significant_trades = df_actions[ (df_actions[TARGET_ASSETS].abs() > 10).any(axis=1) ]
    
    print("\n" + "="*40)
    print(f"      TRADE LOG ({trade_start_date[:4]})      ")
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
    
    print(f"Starting Multi-Asset Backtest (SPY + QQQ)...")
    
    for year in TEST_YEARS:
        print(f"\n=== TESTING YEAR: {year} ===")
        train_start, train_end = "2009-01-01", f"{year}-01-01"
        test_start, test_end = f"{year}-01-01", f"{year+1}-01-01"
        
        ai_return = stock_trading(
            train_start_date=train_start, train_end_date=train_end,
            trade_start_date=test_start, trade_end_date=test_end,
            train_model=True, model_name=f"ppo_momentum_{year}"
        )
        
        spy_loader = RobustYahooDownloader(start_date=test_start, end_date=test_end, ticker_list=["SPY"])
        spy_df = spy_loader.fetch_data()
        spy_return = ((spy_df.iloc[-1]['close'] - spy_df.iloc[0]['close']) / spy_df.iloc[0]['close']) * 100
        
        overall_results.append({
            "Year": year, "Model": f"{ai_return:.2f}%", "SPY": f"{spy_return:.2f}%",
            "Win": "YES" if ai_return > spy_return else "NO"
        })

    print("\n" + pd.DataFrame(overall_results).to_string(index=False))

**ehhhh, last try on this idea, will try new thing after this code**
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
    
    # Create PPO manually to ensure seed is handled correctly
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
    
    # Disable logger to keep output clean
    new_logger = configure(os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}"), ["csv"])
    model_ppo.set_logger(new_logger)
    
    print(f"   > Training Agent (Seed {seed})...")
    # We use the model's own learn method directly
    model_ppo.learn(total_timesteps=60000, tb_log_name=f"{model_name}_{seed}")
    
    return model_ppo
def stock_trading(
    train_start_date: str, train_end_date: str,
    trade_start_date: str, trade_end_date: str,
    train_model: bool = True, 
    model_name: str = "my_ppo_model"
) -> float:
    
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    TARGET_ASSETS = ["SPY", "QQQ", "SHV"]
    
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

    # Signals
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
    
    # Features
    processed['vix_ma'] = processed['vix'].rolling(10).mean()
    processed['fear_signal'] = processed['vix'] / processed['vix_ma']
    processed['rate_ma'] = processed['rate'].rolling(50).mean()
    processed['rate_trend'] = processed['rate'] / processed['rate_ma']
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
    
    my_tech_indicator_list = INDICATORS + ['vix', 'fear_signal', 'rate', 'rate_trend', 'trend_ratio']
    
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
    seeds = [42, 101, 999] # Three different random seeds
    
    if train_model:
        print(f"--- Training Ensemble (3 Agents): {model_name} ---")
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

    # --- ENSEMBLE PREDICTION (VOTING) ---
    print(f"Backtesting with Ensemble Voting...")
    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=None, risk_indicator_col=None, **env_kwargs)
    obs_trade, _ = e_trade_gym.reset()
    done = False
    
    while not done:
        # Get actions from ALL 3 agents
        actions_list = []
        for model in ensemble_models:
            action, _ = model.predict(obs_trade, deterministic=True)
            actions_list.append(action)
        
        # AVERAGE the actions (Consensus Vote)
        # If Agent A says Buy 100, B says Buy 0, C says Sell 50 -> Result: Buy 16
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
    print(f"      ENSEMBLE TRADE LOG ({trade_start_date[:4]})      ")
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
    
    print(f"Starting Ensemble Backtest (SPY + QQQ + SHV)...")
    
    for year in TEST_YEARS:
        print(f"\n=== TESTING YEAR: {year} ===")
        train_start, train_end = "2009-01-01", f"{year}-01-01"
        test_start, test_end = f"{year}-01-01", f"{year+1}-01-01"
        
        ai_return = stock_trading(
            train_start_date=train_start, train_end_date=train_end,
            trade_start_date=test_start, trade_end_date=test_end,
            train_model=True, model_name=f"ppo_ensemble_{year}"
        )
        
        spy_loader = RobustYahooDownloader(start_date=test_start, end_date=test_end, ticker_list=["SPY"])
        spy_df = spy_loader.fetch_data()
        spy_return = ((spy_df.iloc[-1]['close'] - spy_df.iloc[0]['close']) / spy_df.iloc[0]['close']) * 100
        
        overall_results.append({
            "Year": year, "Model": f"{ai_return:.2f}%", "SPY": f"{spy_return:.2f}%",
            "Win": "YES" if ai_return > spy_return else "NO"
        })

    print("\n" + pd.DataFrame(overall_results).to_string(index=False))


**ehh but can run both with SPY and QQQ**
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


**p good**
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


**bad, code before pivot to sales**
from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from stable_baselines3 import PPO
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR, INDICATORS
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# --- CONFIGURATION ---
UNIVERSE = ["SPY", "QQQ", "GLD"] 
MACRO_TICKERS = ["^VIX", "^TNX"] # VIX and 10-Year Treasury Yield

# ----------------------------------------
# 1. ROBUST DOWNLOADER
# ----------------------------------------
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading {self.ticker_list}...")
        try:
            # Threading false to avoid YF rate limits
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date, 
                ignore_tz=True, threads=False, progress=False
            )
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
        
        # Flatten MultiIndex logic
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
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        df['date'] = df['date'].astype(str)
        required = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required if c in df.columns]
        
        df = df[final_cols]
        df = df.drop_duplicates(subset=['date', 'tic']).dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # --- INJECT SYNTHETIC CASH ---
        dates = df.date.unique()
        cash_data = []
        for d in dates:
            cash_data.append([d, 'CASH', 1.0, 1.0, 1.0, 1.0, 0.0])
            
        df_cash = pd.DataFrame(cash_data, columns=['date', 'tic', 'open', 'high', 'low', 'close', 'volume'])
        df_final = pd.concat([df, df_cash], axis=0)
        df_final = df_final.sort_values(['date', 'tic']).reset_index(drop=True)
        return df_final

# ----------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------
def process_data(df):
    print("Processing data with technical indicators...")
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=True, user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    
    # Macro Data Injection
    print(f"Fetching Macro Data: {MACRO_TICKERS}...")
    try:
        macro_raw = yf.download(MACRO_TICKERS, start=df.date.min(), end=df.date.max(), ignore_tz=True, progress=False)
        macro_df = macro_raw['Close'].reset_index()
        macro_df['date'] = macro_df['Date'].astype(str)
        
        if '^VIX' in macro_df.columns: macro_df = macro_df.rename(columns={'^VIX': 'vix'})
        if '^TNX' in macro_df.columns: macro_df = macro_df.rename(columns={'^TNX': 'tnx'})
        
        cols_to_keep = ['date'] + [c for c in ['vix', 'tnx'] if c in macro_df.columns]
        macro_df = macro_df[cols_to_keep]
        
        processed = processed.merge(macro_df, on='date', how='left').ffill().bfill()
    except Exception as e:
        print(f"Macro download failed ({e}), using defaults.")
        processed['vix'] = 20.0
        processed['tnx'] = 3.0

    # Neutralize CASH indicators
    print("Neutralizing CASH technicals...")
    mask = processed['tic'] == 'CASH'
    for col in processed.columns:
        if 'rsi' in col: processed.loc[mask, col] = 50.0 
        elif 'macd' in col: processed.loc[mask, col] = 0.0
        elif 'cci' in col: processed.loc[mask, col] = 0.0
        elif 'adx' in col: processed.loc[mask, col] = 0.0

    processed = processed.fillna(0)
    
    # --- CRITICAL FIX: Ensure Index Alignment ---
    # We factorize the date so Day 0 is index 0 for all stocks, Day 1 is index 1, etc.
    processed.index = processed.date.factorize()[0]
    
    return processed

# ----------------------------------------
# 3. TRAINING SUB-AGENTS (The "Specialists")
# ----------------------------------------
def train_specialist(name, train_start, train_end, processed_data, total_timesteps=20000, forbidden_tickers=None):
    print(f"\n--- Training {name} Agent ({train_start} to {train_end}) ---")
    
    train_data = data_split(processed_data, train_start, train_end)
    stock_dim = len(train_data.tic.unique())
    unique_tics = train_data.tic.unique().tolist()
    
    # --- THE FEE HACK (CONSTRAINT TRAINING) ---
    # If a ticker is forbidden, we set its transaction cost to 50% (0.5)
    # The agent will learn to never touch it.
    buy_cost_list = []
    if forbidden_tickers:
        print(f"Applying HIGH FEES to: {forbidden_tickers}")
        for tic in unique_tics:
            if tic in forbidden_tickers:
                buy_cost_list.append(0.5) # 50% fee (Toxic Asset)
            else:
                buy_cost_list.append(0.001) # Standard 0.1% fee
    else:
        buy_cost_list = [0.001] * stock_dim

    tech_list = INDICATORS + ['vix', 'tnx', 'turbulence']
    
    env_kwargs = {
        "hmax": 100, "initial_amount": 100000, 
        "num_stock_shares": [0]*stock_dim,
        "buy_cost_pct": buy_cost_list, 
        "sell_cost_pct": [0.001]*stock_dim, 
        "state_space": 1 + 2*stock_dim + len(tech_list)*stock_dim, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", model_kwargs={"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64})
    
    trained_model = agent.train_model(model=model, tb_log_name=name, total_timesteps=total_timesteps)
    trained_model.save(os.path.join(TRAINED_MODEL_DIR, name))
    return trained_model

# ----------------------------------------
# 4. HIERARCHICAL MANAGER ENV (Corrected)
# ----------------------------------------
class ManagerEnv(gym.Env):
    def __init__(self, df, sub_agents, env_kwargs, switching_cost_penalty=0.0):
        super(ManagerEnv, self).__init__()
        
        self.df = df.copy()
        # Ensure index aligns with StockTradingEnv expectations (Day 0, Day 0... Day 1, Day 1...)
        self.df.index = self.df.date.factorize()[0]
        
        self.bull_model = sub_agents['bull']
        self.bear_model = sub_agents['bear']
        self.stock_env = StockTradingEnv(df=self.df, **env_kwargs)
        self.action_space = gym.spaces.Discrete(2) # 0: Bull, 1: Bear
        
        # --- MANAGER DASHBOARD ---
        # 4 Inputs: [Val_Norm, VIX, Turb, Trend_Signal]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.last_asset_value = env_kwargs['initial_amount']
        self.prev_action = 0 
        self.switching_cost_penalty = switching_cost_penalty
        self.initial_amount = env_kwargs['initial_amount']

        # Pre-calculate SPY SMA200 for Trend Signal (Robust map)
        spy_df = self.df[self.df['tic'] == 'SPY'].copy()
        if not spy_df.empty:
            spy_df['sma200'] = spy_df['close'].rolling(200).mean()
            self.trend_map = spy_df.set_index('date')['sma200'].to_dict()
        else:
            self.trend_map = {}

    def _get_manager_obs(self):
        current_day = self.stock_env.day
        idx = current_day * self.stock_env.stock_dim
        if idx >= len(self.stock_env.df): idx = len(self.stock_env.df) - 1
        
        current_row = self.stock_env.df.iloc[idx]
        current_date = current_row['date']
        
        # 1. Account Status
        cur_value = self.stock_env.asset_memory[-1]
        val_norm = cur_value / self.initial_amount
        
        # 2. Market Vitals
        raw_vix = current_row['vix'] if 'vix' in current_row else 20.0
        vix = raw_vix / 100.0 
        
        raw_turb = current_row['turbulence'] if 'turbulence' in current_row else 0.0
        turb = raw_turb / 100.0
        
        # 3. Trend Signal (Price vs SMA200)
        trend_signal = 0.0
        if current_date in self.trend_map:
            sma = self.trend_map[current_date]
            spy_row = self.stock_env.df[(self.stock_env.df.date == current_date) & (self.stock_env.df.tic == 'SPY')]
            if not spy_row.empty and not np.isnan(sma):
                price = spy_row['close'].values[0]
                trend_signal = (price - sma) / sma

        return np.array([val_norm, vix, turb, trend_signal], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.stock_env.reset(seed=seed, options=options)
        self.last_asset_value = self.stock_env.asset_memory[-1]
        self.prev_action = 0 
        return self._get_manager_obs(), {}
    
    def step(self, action):
        sub_agent_obs = np.array(self.stock_env.state)
        
        # 1. Select Sub-Agent
        if action == 0:
            sub_action, _ = self.bull_model.predict(sub_agent_obs, deterministic=True)
        else:
            sub_action, _ = self.bear_model.predict(sub_agent_obs, deterministic=True)
            
        # 2. Execute
        _, _, done, truncated, info = self.stock_env.step(sub_action)
        
        # 3. Reward Calculation (The "Pain Penalty")
        current_value = self.stock_env.asset_memory[-1]
        step_return = (current_value - self.last_asset_value) / self.last_asset_value
        self.last_asset_value = current_value
        
        if step_return < 0:
            reward = step_return * 20.0 # 20x Penalty for losing money
        else:
            reward = step_return * 1.0
            
        if action != self.prev_action:
            reward -= self.switching_cost_penalty
            
        self.prev_action = action
        return self._get_manager_obs(), reward, done, truncated, info

def train_manager(train_start, train_end, processed_data, env_kwargs):
    print(f"\n--- Training Master Agent (HRL) ({train_start} to {train_end}) ---")
    
    bull_model = PPO.load(os.path.join(TRAINED_MODEL_DIR, "bull_agent"))
    bear_model = PPO.load(os.path.join(TRAINED_MODEL_DIR, "bear_agent"))
    sub_agents = {'bull': bull_model, 'bear': bear_model}
    
    train_data = data_split(processed_data, train_start, train_end)
    manager_env = ManagerEnv(train_data, sub_agents, env_kwargs, switching_cost_penalty=0.0005)
    
    model = PPO("MlpPolicy", manager_env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=15000)
    
    model.save(os.path.join(TRAINED_MODEL_DIR, "master_agent"))
    return model

# ----------------------------------------
# 5. EXECUTION & BACKTEST
# ----------------------------------------
def run_hrl_backtest(trade_start, trade_end, processed_data, env_kwargs):
    print(f"\n--- HRL Backtest for {trade_start} to {trade_end} ---")
    
    bull_model = PPO.load(os.path.join(TRAINED_MODEL_DIR, "bull_agent"))
    bear_model = PPO.load(os.path.join(TRAINED_MODEL_DIR, "bear_agent"))
    master_model = PPO.load(os.path.join(TRAINED_MODEL_DIR, "master_agent"))
    
    trade_data = data_split(processed_data, trade_start, trade_end)
    stock_env = StockTradingEnv(df=trade_data, **env_kwargs)
    
    # Pre-calc Trend Map for Backtest
    spy_df = trade_data[trade_data['tic'] == 'SPY'].copy()
    trend_map = {}
    if not spy_df.empty:
        spy_df['sma200'] = spy_df['close'].rolling(200).mean()
        trend_map = spy_df.set_index('date')['sma200'].to_dict()

    obs, _ = stock_env.reset()
    done = False
    history_manager = []
    initial_amount = env_kwargs['initial_amount']
    
    while not done:
        # --- CONSTRUCT MANAGER OBSERVATION MANUALLY ---
        current_day = stock_env.day
        idx = current_day * stock_env.stock_dim
        if idx >= len(stock_env.df): idx = len(stock_env.df) - 1
        current_row = stock_env.df.iloc[idx]
        current_date = current_row['date']
        
        cur_value = stock_env.asset_memory[-1]
        val_norm = cur_value / initial_amount
        vix = current_row['vix'] if 'vix' in current_row else 20.0
        turb = current_row['turbulence'] if 'turbulence' in current_row else 0.0
        
        trend_signal = 0.0
        if current_date in trend_map:
            sma = trend_map[current_date]
            spy_row = stock_env.df[(stock_env.df.date == current_date) & (stock_env.df.tic == 'SPY')]
            if not spy_row.empty and not np.isnan(sma):
                price = spy_row['close'].values[0]
                trend_signal = (price - sma) / sma
        
        manager_obs = np.array([val_norm, vix, turb, trend_signal], dtype=np.float32)
        # ----------------------------------------------

        # 1. Get Manager Decision
        manager_action, _ = master_model.predict(manager_obs, deterministic=True)
        
        # 2. Execute Sub-Agent
        if manager_action == 0:
            action, _ = bull_model.predict(obs, deterministic=True)
            history_manager.append("BULL")
        else:
            action, _ = bear_model.predict(obs, deterministic=True)
            history_manager.append("BEAR")
            
        obs, _, dones, _, _ = stock_env.step(action)
        done = dones if isinstance(dones, bool) else dones[0]

    df_account = stock_env.save_asset_memory()
    final_val = df_account['account_value'].iloc[-1]
    init_val = df_account['account_value'].iloc[0]
    return_pct = ((final_val - init_val) / init_val) * 100
    
    bull_count = history_manager.count("BULL")
    bear_count = history_manager.count("BEAR")
    print(f"Manager Choice Distribution: Bull {bull_count} days, Bear {bear_count} days.")
    
    return return_pct

if __name__ == "__main__":
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # 1. Fetch & Process
    print("Fetching Global Data...")
    loader = RobustYahooDownloader(start_date="2007-01-01", end_date="2024-01-01", ticker_list=UNIVERSE)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    # 2. Train Sub-Agents (The Skills)
    
    # A. BULL AGENT (Normal World)
    # Allowed to trade everything normally.
    train_specialist("bull_agent", "2012-01-01", "2019-01-01", df_processed, forbidden_tickers=[])
    
    # B. BEAR AGENT (Constraint Training / Fee Hack)
    # We apply 50% Fees to SPY and QQQ. It learns they are "Toxic".
    print("Training Bear Agent with HIGH FEES on SPY/QQQ...")
    train_specialist("bear_agent", "2008-01-01", "2009-06-01", df_processed, forbidden_tickers=["SPY", "QQQ"])
    
    # 3. Train Master Agent (Real World - Normal Fees)
    # We define the environment with STANDARD fees (0.1%) for the Manager and Backtest.
    stock_dim = len(df_processed.tic.unique())
    tech_list = INDICATORS + ['vix', 'tnx', 'turbulence']
    
    normal_env_kwargs = {
        "hmax": 100, "initial_amount": 100000, 
        "num_stock_shares": [0]*stock_dim,
        "buy_cost_pct": [0.001]*stock_dim, # <--- NORMAL FEES
        "sell_cost_pct": [0.001]*stock_dim, 
        "state_space": 1 + 2*stock_dim + len(tech_list)*stock_dim, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }

    print("Training Manager on Crisis Data (2008-2012)...")
    train_manager("2008-01-01", "2012-01-01", df_processed, normal_env_kwargs)

    # 4. Backtest (The Exam)
    test_years = [2021, 2022, 2023]
    results = []
    
    for year in test_years:
        print(f"\n=== TESTING HIERARCHICAL AGENT ON YEAR {year} ===")
        start = f"{year}-01-01"
        end = f"{year+1}-01-01"
        
        try:
            ai_return = run_hrl_backtest(start, end, df_processed, normal_env_kwargs)
            
            # SPY Benchmark
            spy_subset = df_processed[(df_processed.date >= start) & (df_processed.date < end) & (df_processed.tic == "SPY")]
            if not spy_subset.empty:
                spy_start = spy_subset['close'].iloc[0]
                spy_end = spy_subset['close'].iloc[-1]
                spy_return = ((spy_end - spy_start) / spy_start) * 100
            else:
                spy_return = 0.0
            
            results.append({"Year": year, "HRL_Agent": f"{ai_return:.2f}%", "SPY": f"{spy_return:.2f}%"})
        except Exception as e:
            print(f"Error testing year {year}: {e}")
            import traceback
            traceback.print_exc()

    print("\nFINAL HRL RESULTS:")
    print(pd.DataFrame(results).to_string(index=False))

**biotech p good**
from __future__ import annotations
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
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# --- CONFIGURATION: THE "MAD SCIENCE" BIOTECH PORTFOLIO ---
UNIVERSE = ["XBI", "LABU", "LABD", "VRTX", "SHY"] 

# ----------------------------------------
# 1. ROBUST DOWNLOADER
# ----------------------------------------
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading {self.ticker_list}...")
        try:
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date, 
                ignore_tz=True, threads=False, progress=False
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
            df = df.rename(columns={'Date': 'date', 'Index': 'date'})
            if 'tic' not in df.columns:
                df['tic'] = self.ticker_list[0]

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        df['date'] = df['date'].astype(str)
        required = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required if c in df.columns]
        
        df = df[final_cols]
        df = df.drop_duplicates(subset=['date', 'tic']).dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        return df

# ----------------------------------------
# 2. FEATURE ENGINEERING (FIXED: NO LOOK-AHEAD BIAS)
# ----------------------------------------
def process_data(df):
    print("Processing data with technical indicators...")
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=True, user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    processed = processed.fillna(0)
    
    # 1. Custom Feature: "Panic Factor"
    print("Generating Custom Factors...")
    processed['log_ret'] = np.log(processed['close'] / processed['close'].shift(1)).fillna(0)
    processed['volatility_21'] = processed.groupby('tic')['log_ret'].transform(lambda x: x.rolling(21).std())
    
    # 2. NEW: Market Regime Detection (FIXED)
    print("Detecting Market Regime (With 1-Day Lag)...")
    
    # Extract XBI data to calculate trend
    xbi_df = processed[processed['tic'] == 'XBI'].copy()
    xbi_df['sma_50'] = xbi_df['close'].rolling(50).mean()
    
    # Calculate the raw signal (1 if Bull, 0 if Bear)
    xbi_df['raw_signal'] = np.where(xbi_df['close'] > xbi_df['sma_50'], 1.0, 0.0)
    
    # CRITICAL FIX: Shift by 1 day so we don't peek at today's close
    xbi_df['bull_market'] = xbi_df['raw_signal'].shift(1)
    
    # Merge this signal back so EVERY asset knows yesterday's regime
    processed = processed.merge(xbi_df[['date', 'bull_market']], on='date', how='left')
    
    # Fill NaN (first 50 days) with 0 (assume bear market for safety)
    processed['bull_market'] = processed['bull_market'].fillna(0)
    processed = processed.fillna(0)
    
    processed.index = processed.date.factorize()[0]
    return processed

# ----------------------------------------
# 3. THE SUB-AGENTS (FIXED: HIGHER COSTS)
# ----------------------------------------
def get_agent_params(role):
    if role == "SURFER": 
        return {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0002, "batch_size": 128}
    elif role == "SNIPER": 
        return {"n_steps": 1024, "ent_coef": 0.05, "learning_rate": 0.0005, "batch_size": 64}
    elif role == "SENTINEL": 
        return {"n_steps": 4096, "ent_coef": 0.01, "learning_rate": 0.0001, "batch_size": 256}
    return {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64}

def train_specialist(role, train_data, total_timesteps=20000):
    print(f"Training {role} Agent...")
    
    # 1. Dynamic definitions
    stock_dim = len(train_data.tic.unique())
    my_tech_list = INDICATORS + ['volatility_21', 'bull_market']
    
    # 2. Calculate State Space accurately
    state_space = 1 + 2*stock_dim + (len(my_tech_list) * stock_dim)
    
    print(f"Agent State Space: {state_space} (Stocks: {stock_dim})")
    
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 100000, 
        "num_stock_shares": [0] * stock_dim,
        # CRITICAL FIX: Increased cost to 0.25% to simulate slippage
        "buy_cost_pct": [0.0025] * stock_dim, 
        "sell_cost_pct": [0.0025] * stock_dim, 
        "state_space": state_space, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": my_tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    params = get_agent_params(role)
    model = PPO("MlpPolicy", env_train, verbose=0, **params)
    model.learn(total_timesteps=total_timesteps)
    
    return model, env_kwargs

# ----------------------------------------
# 4. SOFT-VOTING MANAGER
# ----------------------------------------
class SoftVotingManagerEnv(gym.Env):
    def __init__(self, df, sub_agents, env_kwargs):
        super(SoftVotingManagerEnv, self).__init__()
        
        self.df = df.copy()
        self.df.index = self.df.date.factorize()[0]
        
        self.surfer = sub_agents['SURFER']
        self.sniper = sub_agents['SNIPER']
        self.sentinel = sub_agents['SENTINEL']
        
        self.stock_env = StockTradingEnv(df=self.df, **env_kwargs)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.stock_env.observation_space
        
    def reset(self, seed=None, options=None):
        return self.stock_env.reset(seed=seed, options=options)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def step(self, action):
        # 1. Calculate Base Weights
        weights = self.softmax(action) 
        
        # --- REGIME OVERRIDE ---
        current_idx = self.stock_env.day * self.stock_env.stock_dim
        if current_idx >= len(self.stock_env.df):
            current_idx = len(self.stock_env.df) - 1
            
        # This now reads the SHIFTED signal (Yesterday's trend)
        is_bull = self.stock_env.df.iloc[current_idx]['bull_market']
        
        if is_bull > 0.5:
             # If yesterday was Bull, suppress Sentinel today
             weights[2] = weights[2] * 0.2
             weights = weights / np.sum(weights)

        # 2. Get Opinions
        current_obs = self.stock_env.state
        current_obs = np.array(current_obs).reshape(1, -1)
        
        act_surfer, _ = self.surfer.predict(current_obs, deterministic=True)
        act_sniper, _ = self.sniper.predict(current_obs, deterministic=True)
        act_sentinel, _ = self.sentinel.predict(current_obs, deterministic=True)
        
        # 3. Blend Actions
        weighted_action = (act_surfer * weights[0]) + \
                          (act_sniper * weights[1]) + \
                          (act_sentinel * weights[2])
        
        # 4. Execute
        obs, rewards, done, truncated, info = self.stock_env.step(weighted_action[0])
        return obs, rewards, done, truncated, info

# ----------------------------------------
# 5. WALK-FORWARD EXECUTION LOOP
# ----------------------------------------
def run_mad_science_experiment():
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    loader = RobustYahooDownloader(start_date="2010-01-01", end_date="2024-01-01", ticker_list=UNIVERSE)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    train_window = 5
    unique_years = sorted(pd.to_datetime(df_processed['date']).dt.year.unique())
    
    results_log = []
    
    start_idx = 0
    while start_idx + train_window < len(unique_years):
        train_years = unique_years[start_idx : start_idx + train_window]
        test_year = unique_years[start_idx + train_window]
        
        print(f"\n==================================================")
        print(f"WINDOW: Train {min(train_years)}-{max(train_years)} | Test {test_year}")
        print(f"==================================================")
        
        train_mask = pd.to_datetime(df_processed['date']).dt.year.isin(train_years)
        test_mask = pd.to_datetime(df_processed['date']).dt.year == test_year
        
        train_df = df_processed[train_mask].copy()
        test_df = df_processed[test_mask].copy()
        
        train_df.index = train_df.date.factorize()[0]
        test_df.index = test_df.date.factorize()[0]
        
        surfer, env_conf = train_specialist("SURFER", train_df, total_timesteps=15000)
        sniper, _ = train_specialist("SNIPER", train_df, total_timesteps=15000)
        sentinel, _ = train_specialist("SENTINEL", train_df, total_timesteps=15000)
        
        sub_agents = {"SURFER": surfer, "SNIPER": sniper, "SENTINEL": sentinel}
        
        print("Training Soft-Voting Manager...")
        manager_env = SoftVotingManagerEnv(train_df, sub_agents, env_conf)
        manager_env_vec = DummyVecEnv([lambda: manager_env])
        
        manager_model = PPO("MlpPolicy", manager_env_vec, verbose=0, learning_rate=0.0003)
        manager_model.learn(total_timesteps=10000)
        
        print(f"Testing on {test_year}...")
        test_env = SoftVotingManagerEnv(test_df, sub_agents, env_conf)
        
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = manager_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = test_env.step(action)
            
        final_asset = test_env.stock_env.asset_memory[-1]
        initial = env_conf['initial_amount']
        roi = ((final_asset - initial) / initial) * 100
        
        print(f"--> Year {test_year} ROI: {roi:.2f}%")
        results_log.append({"Year": test_year, "ROI": roi})
        
        start_idx += 1

    print("\n--- FINAL MAD SCIENCE RESULTS ---")
    print(pd.DataFrame(results_log))

if __name__ == "__main__":
    run_mad_science_experiment()




**good version biotech trial w tests in comments**
from __future__ import annotations
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
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# --- CONFIGURATION: THE "MAD SCIENCE" BIOTECH PORTFOLIO ---
UNIVERSE = ["XBI", "LABU", "LABD", "VRTX", "SHY"] 

# ----------------------------------------
# 1. ROBUST DOWNLOADER
# ----------------------------------------
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading {self.ticker_list}...")
        try:
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date, 
                ignore_tz=True, threads=False, progress=False
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
            df = df.rename(columns={'Date': 'date', 'Index': 'date'})
            if 'tic' not in df.columns:
                df['tic'] = self.ticker_list[0]

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        df['date'] = df['date'].astype(str)
        required = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required if c in df.columns]
        
        df = df[final_cols]
        df = df.drop_duplicates(subset=['date', 'tic']).dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        return df

# ----------------------------------------
# 2. FEATURE ENGINEERING (FIXED: NO LOOK-AHEAD BIAS)
# ----------------------------------------
def process_data(df):
    print("Processing data with technical indicators...")
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=True, user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    processed = processed.fillna(0)
    
    # 1. Custom Feature: "Panic Factor"
    print("Generating Custom Factors...")
    processed['log_ret'] = np.log(processed['close'] / processed['close'].shift(1)).fillna(0)
    processed['volatility_21'] = processed.groupby('tic')['log_ret'].transform(lambda x: x.rolling(21).std())
    
    # 2. NEW: Market Regime Detection (FIXED)
    print("Detecting Market Regime (With 1-Day Lag)...")
    
    # Extract XBI data to calculate trend
    xbi_df = processed[processed['tic'] == 'XBI'].copy()
    xbi_df['sma_50'] = xbi_df['close'].rolling(50).mean()
    
    # Calculate the raw signal (1 if Bull, 0 if Bear)
    xbi_df['raw_signal'] = np.where(xbi_df['close'] > xbi_df['sma_50'], 1.0, 0.0)
    
    # CRITICAL FIX: Shift by 1 day so we don't peek at today's close
    xbi_df['bull_market'] = xbi_df['raw_signal'].shift(1)
    
    # Merge this signal back so EVERY asset knows yesterday's regime
    processed = processed.merge(xbi_df[['date', 'bull_market']], on='date', how='left')
    
    # Fill NaN (first 50 days) with 0 (assume bear market for safety)
    processed['bull_market'] = processed['bull_market'].fillna(0)
    processed = processed.fillna(0)
    
    processed.index = processed.date.factorize()[0]
    return processed

# ----------------------------------------
# 3. THE SUB-AGENTS (FIXED: HIGHER COSTS)
# ----------------------------------------
def get_agent_params(role):
    if role == "SURFER": 
        return {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0002, "batch_size": 128}
    elif role == "SNIPER": 
        return {"n_steps": 1024, "ent_coef": 0.05, "learning_rate": 0.0005, "batch_size": 64}
    elif role == "SENTINEL": 
        return {"n_steps": 4096, "ent_coef": 0.01, "learning_rate": 0.0001, "batch_size": 256}
    return {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64}

def train_specialist(role, train_data, total_timesteps=20000):
    print(f"Training {role} Agent...")
    
    # 1. Dynamic definitions
    stock_dim = len(train_data.tic.unique())
    my_tech_list = INDICATORS + ['volatility_21', 'bull_market']
    
    # 2. Calculate State Space accurately
    state_space = 1 + 2*stock_dim + (len(my_tech_list) * stock_dim)
    
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 100000, 
        "num_stock_shares": [0] * stock_dim,
        # CRITICAL FIX: Increased cost to 0.25% to simulate slippage
        "buy_cost_pct": [0.0025] * stock_dim, 
        "sell_cost_pct": [0.0025] * stock_dim, 
        "state_space": state_space, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": my_tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    params = get_agent_params(role)
    model = PPO("MlpPolicy", env_train, verbose=0, **params)
    model.learn(total_timesteps=total_timesteps)
    
    return model, env_kwargs

# ----------------------------------------
# 4. SOFT-VOTING MANAGER
# ----------------------------------------
class SoftVotingManagerEnv(gym.Env):
    def __init__(self, df, sub_agents, env_kwargs):
        super(SoftVotingManagerEnv, self).__init__()
        
        self.df = df.copy()
        self.df.index = self.df.date.factorize()[0]

        # Handle Missing Agents (For Ablation Testing)
        self.surfer = sub_agents.get('SURFER', None)
        self.sniper = sub_agents.get('SNIPER', None)
        self.sentinel = sub_agents.get('SENTINEL', None)
        
        self.stock_env = StockTradingEnv(df=self.df, **env_kwargs)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.stock_env.observation_space
        
    def reset(self, seed=None, options=None):
        return self.stock_env.reset(seed=seed, options=options)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def step(self, action):
        # 1. Calculate Base Weights
        weights = self.softmax(action) 
        
        # --- REGIME OVERRIDE ---
        current_idx = self.stock_env.day * self.stock_env.stock_dim
        if current_idx >= len(self.stock_env.df):
            current_idx = len(self.stock_env.df) - 1
            
        # This now reads the SHIFTED signal (Yesterday's trend)
        is_bull = self.stock_env.df.iloc[current_idx]['bull_market']
        
        if is_bull > 0.5:
             # If yesterday was Bull, suppress Sentinel today
             weights[2] = weights[2] * 0.2
             weights = weights / np.sum(weights)

        # 2. Get Opinions
        current_obs = self.stock_env.state
        current_obs = np.array(current_obs).reshape(1, -1)

        # --- FIX: Safe checks for ALL agents (Prevents NameError) ---
        if self.surfer:
             act_surfer, _ = self.surfer.predict(current_obs, deterministic=True)
        else:
             act_surfer = np.zeros(self.stock_env.action_space.shape)

        if self.sniper:
             act_sniper, _ = self.sniper.predict(current_obs, deterministic=True)
        else:
             act_sniper = np.zeros(self.stock_env.action_space.shape)
             
        if self.sentinel:
             act_sentinel, _ = self.sentinel.predict(current_obs, deterministic=True)
        else:
             act_sentinel = np.zeros(self.stock_env.action_space.shape)
        
        # 3. Blend Actions
        weighted_action = (act_surfer * weights[0]) + \
                          (act_sniper * weights[1]) + \
                          (act_sentinel * weights[2])
        
        # 4. Execute
        obs, rewards, done, truncated, info = self.stock_env.step(weighted_action[0])
        return obs, rewards, done, truncated, info

# ----------------------------------------
# 5. DIAGNOSTIC EXECUTION LOOP
# ----------------------------------------
def run_diagnostic_experiment():
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # --- DIAGNOSTIC SETTINGS ---
    # 1. LEVERAGE CHECK: Set to False to remove LABU/LABD
    USE_LEVERAGE = True  
    
    # 2. ABLATION CHECK: Remove an agent to see if it matters?
    # Options: "NONE" (Full Team), "NO_SNIPER", "NO_SURFER"
    ABLATION_MODE = "NONE" 
    
    # 3. LUCK CHECK: How many seeds to run? (Standard is 3-5)
    SEEDS = [42, 101, 999] 
    # ---------------------------

    # 1. Define Universe based on settings
    if USE_LEVERAGE:
        current_universe = ["XBI", "LABU", "LABD", "VRTX", "SHY"]
        print(f"\n[TEST] Running with FULL LEVERAGE: {current_universe}")
    else:
        current_universe = ["XBI", "VRTX", "SHY"]
        print(f"\n[TEST] Running in BORING MODE (No Leverage): {current_universe}")

    # 2. Fetch Data
    loader = RobustYahooDownloader(start_date="2010-01-01", end_date="2024-01-01", ticker_list=current_universe)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    # Storage for final stats
    all_seeds_results = []

    for seed in SEEDS:
        print(f"\n\n>>> STARTING RUN WITH SEED {seed} <<<")
        
        # Set Random Seeds for Reproducibility
        np.random.seed(seed)
        import random
        random.seed(seed)
        import torch
        torch.manual_seed(seed)
        
        # Walk-Forward Setup
        train_window = 4 
        validation_window = 1
        unique_years = sorted(pd.to_datetime(df_processed['date']).dt.year.unique())
        
        seed_rois = [] # Store ROI for each year in this seed
        
        start_idx = 0
        while start_idx + train_window + validation_window < len(unique_years):
            
            # Define Years
            agent_train_years = unique_years[start_idx : start_idx + train_window]
            manager_train_year = unique_years[start_idx + train_window]
            test_year = unique_years[start_idx + train_window + validation_window]
            
            # Slice Data
            agent_df = df_processed[pd.to_datetime(df_processed['date']).dt.year.isin(agent_train_years)].copy()
            manager_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == manager_train_year].copy()
            test_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == test_year].copy()
            
            agent_df.index = agent_df.date.factorize()[0]
            manager_df.index = manager_df.date.factorize()[0]
            test_df.index = test_df.date.factorize()[0]
            
            # --- TRAIN SPECIALISTS (With Ablation Logic) ---
            sub_agents = {}
            
            # Surfer
            if ABLATION_MODE != "NO_SURFER":
                surfer, env_conf = train_specialist("SURFER", agent_df, total_timesteps=15000)
                sub_agents["SURFER"] = surfer
            
            # Sniper
            if ABLATION_MODE != "NO_SNIPER":
                sniper, _ = train_specialist("SNIPER", agent_df, total_timesteps=15000)
                sub_agents["SNIPER"] = sniper
                
            # Sentinel (Always trained, so we capture env_conf here to avoid crashes)
            sentinel, env_conf = train_specialist("SENTINEL", agent_df, total_timesteps=15000)
            sub_agents["SENTINEL"] = sentinel
            
            # --- TRAIN MANAGER ---
            manager_env = SoftVotingManagerEnv(manager_df, sub_agents, env_conf)
            manager_env_vec = DummyVecEnv([lambda: manager_env])
            
            manager_model = PPO("MlpPolicy", manager_env_vec, verbose=0, learning_rate=0.0003, seed=seed)
            manager_model.learn(total_timesteps=10000)
            
            # --- TEST ---
            test_env = SoftVotingManagerEnv(test_df, sub_agents, env_conf)
            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = manager_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = test_env.step(action)
                
            final_asset = test_env.stock_env.asset_memory[-1]
            initial = env_conf['initial_amount']
            roi = ((final_asset - initial) / initial) * 100
            
            print(f"Seed {seed} | Year {test_year} | ROI: {roi:.2f}%")
            seed_rois.append(roi)
            start_idx += 1
            
        avg_roi = np.mean(seed_rois)
        print(f"--> SEED {seed} AVERAGE ANNUAL ROI: {avg_roi:.2f}%")
        all_seeds_results.append(avg_roi)

    # --- FINAL REPORT ---
    print("\n========================================")
    print("      DIAGNOSTIC FINAL REPORT")
    print("========================================")
    print(f"Settings: Leverage={USE_LEVERAGE}, Ablation={ABLATION_MODE}")
    print(f"Seeds Run: {SEEDS}")
    print(f"Results (Avg ROI per seed): {all_seeds_results}")
    print(f"Mean ROI: {np.mean(all_seeds_results):.2f}%")
    print(f"Std Dev:  {np.std(all_seeds_results):.2f}")
    print("========================================")
    
    if np.std(all_seeds_results) > abs(np.mean(all_seeds_results)):
        print("FAIL: Strategy is UNSTABLE (Std Dev > Mean).")
    else:
        print("PASS: Strategy is ROBUST.")

if __name__ == "__main__":
    run_diagnostic_experiment()


"""
# ----------------------------------------
# 5. DIAGNOSTIC EXECUTION LOOP
# ----------------------------------------
def run_diagnostic_experiment():
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # --- DIAGNOSTIC SETTINGS ---
    # 1. LEVERAGE CHECK: Set to False to remove LABU/LABD
    USE_LEVERAGE = True  
    
    # 2. ABLATION CHECK: Remove an agent to see if it matters?
    # Options: "NONE" (Full Team), "NO_SNIPER", "NO_SURFER"
    ABLATION_MODE = "NONE" 
    
    # 3. LUCK CHECK: How many seeds to run? (Standard is 3-5)
    SEEDS = [42, 101, 999] 
    # ---------------------------

    # 1. Define Universe based on settings
    if USE_LEVERAGE:
        current_universe = ["XBI", "LABU", "LABD", "VRTX", "SHY"]
        print(f"\n[TEST] Running with FULL LEVERAGE: {current_universe}")
    else:
        current_universe = ["XBI", "VRTX", "SHY"]
        print(f"\n[TEST] Running in BORING MODE (No Leverage): {current_universe}")

    # 2. Fetch Data
    loader = RobustYahooDownloader(start_date="2010-01-01", end_date="2024-01-01", ticker_list=current_universe)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    # Storage for final stats
    all_seeds_results = []

    for seed in SEEDS:
        print(f"\n\n>>> STARTING RUN WITH SEED {seed} <<<")
        
        # Set Random Seeds for Reproducibility
        np.random.seed(seed)
        import random
        random.seed(seed)
        import torch
        torch.manual_seed(seed)
        
        # Walk-Forward Setup
        train_window = 4 
        validation_window = 1
        unique_years = sorted(pd.to_datetime(df_processed['date']).dt.year.unique())
        
        seed_rois = [] # Store ROI for each year in this seed
        
        start_idx = 0
        while start_idx + train_window + validation_window < len(unique_years):
            
            # Define Years
            agent_train_years = unique_years[start_idx : start_idx + train_window]
            manager_train_year = unique_years[start_idx + train_window]
            test_year = unique_years[start_idx + train_window + validation_window]
            
            # Slice Data
            agent_df = df_processed[pd.to_datetime(df_processed['date']).dt.year.isin(agent_train_years)].copy()
            manager_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == manager_train_year].copy()
            test_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == test_year].copy()
            
            agent_df.index = agent_df.date.factorize()[0]
            manager_df.index = manager_df.date.factorize()[0]
            test_df.index = test_df.date.factorize()[0]
            
            # --- TRAIN SPECIALISTS (With Ablation Logic) ---
            sub_agents = {}
            
            # Surfer
            if ABLATION_MODE != "NO_SURFER":
                surfer, env_conf = train_specialist("SURFER", agent_df, total_timesteps=15000)
                sub_agents["SURFER"] = surfer
            
            # Sniper
            if ABLATION_MODE != "NO_SNIPER":
                sniper, _ = train_specialist("SNIPER", agent_df, total_timesteps=15000)
                sub_agents["SNIPER"] = sniper
                
            # Sentinel (Always keep Sentinel for safety, or test removing it too)
            sentinel, env_conf = train_specialist("SENTINEL", agent_df, total_timesteps=15000)
            sub_agents["SENTINEL"] = sentinel
            
            # --- TRAIN MANAGER ---
            # Note: You might need to adjust SoftVotingManagerEnv to handle missing keys if you remove agents
            # For simplicity, we assume the class handles it or we pass a dummy
            
            manager_env = SoftVotingManagerEnv(manager_df, sub_agents, env_conf)
            manager_env_vec = DummyVecEnv([lambda: manager_env])
            
            manager_model = PPO("MlpPolicy", manager_env_vec, verbose=0, learning_rate=0.0003, seed=seed)
            manager_model.learn(total_timesteps=10000)
            
            # --- TEST ---
            test_env = SoftVotingManagerEnv(test_df, sub_agents, env_conf)
            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = manager_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = test_env.step(action)
                
            final_asset = test_env.stock_env.asset_memory[-1]
            initial = env_conf['initial_amount']
            roi = ((final_asset - initial) / initial) * 100
            
            print(f"Seed {seed} | Year {test_year} | ROI: {roi:.2f}%")
            seed_rois.append(roi)
            start_idx += 1
            
        avg_roi = np.mean(seed_rois)
        print(f"--> SEED {seed} AVERAGE ANNUAL ROI: {avg_roi:.2f}%")
        all_seeds_results.append(avg_roi)

    # --- FINAL REPORT ---
    print("\n========================================")
    print("      DIAGNOSTIC FINAL REPORT")
    print("========================================")
    print(f"Settings: Leverage={USE_LEVERAGE}, Ablation={ABLATION_MODE}")
    print(f"Seeds Run: {SEEDS}")
    print(f"Results (Avg ROI per seed): {all_seeds_results}")
    print(f"Mean ROI: {np.mean(all_seeds_results):.2f}%")
    print(f"Std Dev:  {np.std(all_seeds_results):.2f}")
    print("========================================")
    
    if np.std(all_seeds_results) > abs(np.mean(all_seeds_results)):
        print("FAIL: Strategy is UNSTABLE (Std Dev > Mean).")
    else:
        print("PASS: Strategy is ROBUST.")

if __name__ == "__main__":
    run_diagnostic_experiment()
# ----------------------------------------
# 5. WALK-FORWARD EXECUTION LOOP
# ----------------------------------------
# ----------------------------------------
# 5. WALK-FORWARD EXECUTION LOOP (FIXED: VALIDATION SPLIT)
# ----------------------------------------
"""


**last saved before 15 yr backtest**
from __future__ import annotations
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
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# --- CONFIGURATION: THE "MAD SCIENCE" BIOTECH PORTFOLIO ---
UNIVERSE = ["XBI", "LABU", "LABD", "VRTX", "SHY"] 

# ----------------------------------------
# 1. ROBUST DOWNLOADER
# ----------------------------------------
class RobustYahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        print(f"Downloading {self.ticker_list}...")
        try:
            raw_df = yf.download(
                self.ticker_list, start=self.start_date, end=self.end_date, 
                ignore_tz=True, threads=False, progress=False
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
            df = df.rename(columns={'Date': 'date', 'Index': 'date'})
            if 'tic' not in df.columns:
                df['tic'] = self.ticker_list[0]

        df.columns = df.columns.str.lower()
        if 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
            
        df['date'] = df['date'].astype(str)
        required = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required if c in df.columns]
        
        df = df[final_cols]
        df = df.drop_duplicates(subset=['date', 'tic']).dropna()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        return df

# ----------------------------------------
# 2. FEATURE ENGINEERING (FIXED: NO LOOK-AHEAD BIAS)
# ----------------------------------------
def process_data(df):
    print("Processing data with technical indicators...")
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False, use_turbulence=True, user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    processed = processed.fillna(0)
    
    # 1. Custom Feature: "Panic Factor"
    print("Generating Custom Factors...")
    processed['log_ret'] = np.log(processed['close'] / processed['close'].shift(1)).fillna(0)
    processed['volatility_21'] = processed.groupby('tic')['log_ret'].transform(lambda x: x.rolling(21).std())
    
    # 2. NEW: Market Regime Detection (FIXED)
    print("Detecting Market Regime (With 1-Day Lag)...")
    
    # Extract XBI data to calculate trend
    xbi_df = processed[processed['tic'] == 'XBI'].copy()
    xbi_df['sma_50'] = xbi_df['close'].rolling(50).mean()
    
    # Calculate the raw signal (1 if Bull, 0 if Bear)
    xbi_df['raw_signal'] = np.where(xbi_df['close'] > xbi_df['sma_50'], 1.0, 0.0)
    
    # CRITICAL FIX: Shift by 1 day so we don't peek at today's close
    xbi_df['bull_market'] = xbi_df['raw_signal'].shift(1)
    
    # Merge this signal back so EVERY asset knows yesterday's regime
    processed = processed.merge(xbi_df[['date', 'bull_market']], on='date', how='left')
    
    # Fill NaN (first 50 days) with 0 (assume bear market for safety)
    processed['bull_market'] = processed['bull_market'].fillna(0)
    processed = processed.fillna(0)
    
    processed.index = processed.date.factorize()[0]
    return processed

# ----------------------------------------
# 3. THE SUB-AGENTS (FIXED: HIGHER COSTS)
# ----------------------------------------
def get_agent_params(role):
    if role == "SURFER": 
        return {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0002, "batch_size": 128}
    elif role == "SNIPER": 
        return {"n_steps": 1024, "ent_coef": 0.05, "learning_rate": 0.0005, "batch_size": 64}
    elif role == "SENTINEL": 
        return {"n_steps": 4096, "ent_coef": 0.01, "learning_rate": 0.0001, "batch_size": 256}
    return {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64}

def train_specialist(role, train_data, total_timesteps=20000):
    print(f"Training {role} Agent...")
    
    # 1. Dynamic definitions
    stock_dim = len(train_data.tic.unique())
    my_tech_list = INDICATORS + ['volatility_21', 'bull_market']
    
    # 2. Calculate State Space accurately
    state_space = 1 + 2*stock_dim + (len(my_tech_list) * stock_dim)
    
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 100000, 
        "num_stock_shares": [0] * stock_dim,
        # CRITICAL FIX: Increased cost to 0.25% to simulate slippage
        "buy_cost_pct": [0.0025] * stock_dim, 
        "sell_cost_pct": [0.0025] * stock_dim, 
        "state_space": state_space, 
        "stock_dim": stock_dim, 
        "tech_indicator_list": my_tech_list, 
        "action_space": stock_dim, 
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    params = get_agent_params(role)
    model = PPO("MlpPolicy", env_train, verbose=0, **params)
    model.learn(total_timesteps=total_timesteps)
    
    return model, env_kwargs

# ----------------------------------------
# 4. SOFT-VOTING MANAGER
# ----------------------------------------
class SoftVotingManagerEnv(gym.Env):
    def __init__(self, df, sub_agents, env_kwargs):
        super(SoftVotingManagerEnv, self).__init__()
        
        self.df = df.copy()
        self.df.index = self.df.date.factorize()[0]

        # Handle Missing Agents (For Ablation Testing)
        self.surfer = sub_agents.get('SURFER', None)
        self.sniper = sub_agents.get('SNIPER', None)
        self.sentinel = sub_agents.get('SENTINEL', None)
        
        self.stock_env = StockTradingEnv(df=self.df, **env_kwargs)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.stock_env.observation_space
        
    def reset(self, seed=None, options=None):
        return self.stock_env.reset(seed=seed, options=options)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def step(self, action):
        # 1. Calculate Base Weights
        weights = self.softmax(action) 
        
        # --- REGIME OVERRIDE ---
        current_idx = self.stock_env.day * self.stock_env.stock_dim
        if current_idx >= len(self.stock_env.df):
            current_idx = len(self.stock_env.df) - 1
            
        # This now reads the SHIFTED signal (Yesterday's trend)
        is_bull = self.stock_env.df.iloc[current_idx]['bull_market']
        
        if is_bull > 0.5:
             # If yesterday was Bull, suppress Sentinel today
             weights[2] = weights[2] * 0.2
             weights = weights / np.sum(weights)

        # 2. Get Opinions
        current_obs = self.stock_env.state
        current_obs = np.array(current_obs).reshape(1, -1)

        # --- FIX: Safe checks for ALL agents (Prevents NameError) ---
        if self.surfer:
             act_surfer, _ = self.surfer.predict(current_obs, deterministic=True)
        else:
             act_surfer = np.zeros(self.stock_env.action_space.shape)

        if self.sniper:
             act_sniper, _ = self.sniper.predict(current_obs, deterministic=True)
        else:
             act_sniper = np.zeros(self.stock_env.action_space.shape)
             
        if self.sentinel:
             act_sentinel, _ = self.sentinel.predict(current_obs, deterministic=True)
        else:
             act_sentinel = np.zeros(self.stock_env.action_space.shape)
        
        # 3. Blend Actions
        weighted_action = (act_surfer * weights[0]) + \
                          (act_sniper * weights[1]) + \
                          (act_sentinel * weights[2])
        
        # 4. Execute
        obs, rewards, done, truncated, info = self.stock_env.step(weighted_action[0])
        return obs, rewards, done, truncated, info

# ----------------------------------------
# 5. DIAGNOSTIC EXECUTION LOOP
# ----------------------------------------
def run_diagnostic_experiment():
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # --- DIAGNOSTIC SETTINGS ---
    # 1. LEVERAGE CHECK: Set to False to remove LABU/LABD
    USE_LEVERAGE = True  
    
    # 2. ABLATION CHECK: Remove an agent to see if it matters?
    # Options: "NONE" (Full Team), "NO_SNIPER", "NO_SURFER"
    ABLATION_MODE = "NONE" 
    
    # 3. LUCK CHECK: How many seeds to run? (Standard is 3-5)
    SEEDS = [42, 101, 999] 
    # ---------------------------

    # 1. Define Universe based on settings
    if USE_LEVERAGE:
        current_universe = ["XBI", "LABU", "LABD", "VRTX", "SHY"]
        print(f"\n[TEST] Running with FULL LEVERAGE: {current_universe}")
    else:
        current_universe = ["XBI", "VRTX", "SHY"]
        print(f"\n[TEST] Running in BORING MODE (No Leverage): {current_universe}")

    # 2. Fetch Data
    loader = RobustYahooDownloader(start_date="2010-01-01", end_date="2024-01-01", ticker_list=current_universe)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    # Storage for final stats
    all_seeds_results = []

    for seed in SEEDS:
        print(f"\n\n>>> STARTING RUN WITH SEED {seed} <<<")
        
        # Set Random Seeds for Reproducibility
        np.random.seed(seed)
        import random
        random.seed(seed)
        import torch
        torch.manual_seed(seed)
        
        # Walk-Forward Setup
        train_window = 4 
        validation_window = 1
        unique_years = sorted(pd.to_datetime(df_processed['date']).dt.year.unique())
        
        seed_rois = [] # Store ROI for each year in this seed
        
        start_idx = 0
        while start_idx + train_window + validation_window < len(unique_years):
            
            # Define Years
            agent_train_years = unique_years[start_idx : start_idx + train_window]
            manager_train_year = unique_years[start_idx + train_window]
            test_year = unique_years[start_idx + train_window + validation_window]
            
            # Slice Data
            agent_df = df_processed[pd.to_datetime(df_processed['date']).dt.year.isin(agent_train_years)].copy()
            manager_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == manager_train_year].copy()
            test_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == test_year].copy()
            
            agent_df.index = agent_df.date.factorize()[0]
            manager_df.index = manager_df.date.factorize()[0]
            test_df.index = test_df.date.factorize()[0]
            
            # --- TRAIN SPECIALISTS (With Ablation Logic) ---
            sub_agents = {}
            
            # Surfer
            if ABLATION_MODE != "NO_SURFER":
                surfer, env_conf = train_specialist("SURFER", agent_df, total_timesteps=15000)
                sub_agents["SURFER"] = surfer
            
            # Sniper
            if ABLATION_MODE != "NO_SNIPER":
                sniper, _ = train_specialist("SNIPER", agent_df, total_timesteps=15000)
                sub_agents["SNIPER"] = sniper
                
            # Sentinel (Always trained, so we capture env_conf here to avoid crashes)
            sentinel, env_conf = train_specialist("SENTINEL", agent_df, total_timesteps=15000)
            sub_agents["SENTINEL"] = sentinel
            
            # --- TRAIN MANAGER ---
            manager_env = SoftVotingManagerEnv(manager_df, sub_agents, env_conf)
            manager_env_vec = DummyVecEnv([lambda: manager_env])
            
            manager_model = PPO("MlpPolicy", manager_env_vec, verbose=0, learning_rate=0.0003, seed=seed)
            manager_model.learn(total_timesteps=10000)
            
            # --- TEST ---
            test_env = SoftVotingManagerEnv(test_df, sub_agents, env_conf)
            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = manager_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = test_env.step(action)
                
            final_asset = test_env.stock_env.asset_memory[-1]
            initial = env_conf['initial_amount']
            roi = ((final_asset - initial) / initial) * 100
            
            print(f"Seed {seed} | Year {test_year} | ROI: {roi:.2f}%")
            seed_rois.append(roi)
            start_idx += 1
            
        avg_roi = np.mean(seed_rois)
        print(f"--> SEED {seed} AVERAGE ANNUAL ROI: {avg_roi:.2f}%")
        all_seeds_results.append(avg_roi)

    # --- FINAL REPORT ---
    print("\n========================================")
    print("      DIAGNOSTIC FINAL REPORT")
    print("========================================")
    print(f"Settings: Leverage={USE_LEVERAGE}, Ablation={ABLATION_MODE}")
    print(f"Seeds Run: {SEEDS}")
    print(f"Results (Avg ROI per seed): {all_seeds_results}")
    print(f"Mean ROI: {np.mean(all_seeds_results):.2f}%")
    print(f"Std Dev:  {np.std(all_seeds_results):.2f}")
    print("========================================")
    
    if np.std(all_seeds_results) > abs(np.mean(all_seeds_results)):
        print("FAIL: Strategy is UNSTABLE (Std Dev > Mean).")
    else:
        print("PASS: Strategy is ROBUST.")

if __name__ == "__main__":
    run_diagnostic_experiment()


"""
# ----------------------------------------
# 5. DIAGNOSTIC EXECUTION LOOP
# ----------------------------------------
def run_diagnostic_experiment():
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # --- DIAGNOSTIC SETTINGS ---
    # 1. LEVERAGE CHECK: Set to False to remove LABU/LABD
    USE_LEVERAGE = True  
    
    # 2. ABLATION CHECK: Remove an agent to see if it matters?
    # Options: "NONE" (Full Team), "NO_SNIPER", "NO_SURFER"
    ABLATION_MODE = "NONE" 
    
    # 3. LUCK CHECK: How many seeds to run? (Standard is 3-5)
    SEEDS = [42, 101, 999] 
    # ---------------------------

    # 1. Define Universe based on settings
    if USE_LEVERAGE:
        current_universe = ["XBI", "LABU", "LABD", "VRTX", "SHY"]
        print(f"\n[TEST] Running with FULL LEVERAGE: {current_universe}")
    else:
        current_universe = ["XBI", "VRTX", "SHY"]
        print(f"\n[TEST] Running in BORING MODE (No Leverage): {current_universe}")

    # 2. Fetch Data
    loader = RobustYahooDownloader(start_date="2010-01-01", end_date="2024-01-01", ticker_list=current_universe)
    df_raw = loader.fetch_data()
    df_processed = process_data(df_raw)
    
    # Storage for final stats
    all_seeds_results = []

    for seed in SEEDS:
        print(f"\n\n>>> STARTING RUN WITH SEED {seed} <<<")
        
        # Set Random Seeds for Reproducibility
        np.random.seed(seed)
        import random
        random.seed(seed)
        import torch
        torch.manual_seed(seed)
        
        # Walk-Forward Setup
        train_window = 4 
        validation_window = 1
        unique_years = sorted(pd.to_datetime(df_processed['date']).dt.year.unique())
        
        seed_rois = [] # Store ROI for each year in this seed
        
        start_idx = 0
        while start_idx + train_window + validation_window < len(unique_years):
            
            # Define Years
            agent_train_years = unique_years[start_idx : start_idx + train_window]
            manager_train_year = unique_years[start_idx + train_window]
            test_year = unique_years[start_idx + train_window + validation_window]
            
            # Slice Data
            agent_df = df_processed[pd.to_datetime(df_processed['date']).dt.year.isin(agent_train_years)].copy()
            manager_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == manager_train_year].copy()
            test_df = df_processed[pd.to_datetime(df_processed['date']).dt.year == test_year].copy()
            
            agent_df.index = agent_df.date.factorize()[0]
            manager_df.index = manager_df.date.factorize()[0]
            test_df.index = test_df.date.factorize()[0]
            
            # --- TRAIN SPECIALISTS (With Ablation Logic) ---
            sub_agents = {}
            
            # Surfer
            if ABLATION_MODE != "NO_SURFER":
                surfer, env_conf = train_specialist("SURFER", agent_df, total_timesteps=15000)
                sub_agents["SURFER"] = surfer
            
            # Sniper
            if ABLATION_MODE != "NO_SNIPER":
                sniper, _ = train_specialist("SNIPER", agent_df, total_timesteps=15000)
                sub_agents["SNIPER"] = sniper
                
            # Sentinel (Always keep Sentinel for safety, or test removing it too)
            sentinel, env_conf = train_specialist("SENTINEL", agent_df, total_timesteps=15000)
            sub_agents["SENTINEL"] = sentinel
            
            # --- TRAIN MANAGER ---
            # Note: You might need to adjust SoftVotingManagerEnv to handle missing keys if you remove agents
            # For simplicity, we assume the class handles it or we pass a dummy
            
            manager_env = SoftVotingManagerEnv(manager_df, sub_agents, env_conf)
            manager_env_vec = DummyVecEnv([lambda: manager_env])
            
            manager_model = PPO("MlpPolicy", manager_env_vec, verbose=0, learning_rate=0.0003, seed=seed)
            manager_model.learn(total_timesteps=10000)
            
            # --- TEST ---
            test_env = SoftVotingManagerEnv(test_df, sub_agents, env_conf)
            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = manager_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = test_env.step(action)
                
            final_asset = test_env.stock_env.asset_memory[-1]
            initial = env_conf['initial_amount']
            roi = ((final_asset - initial) / initial) * 100
            
            print(f"Seed {seed} | Year {test_year} | ROI: {roi:.2f}%")
            seed_rois.append(roi)
            start_idx += 1
            
        avg_roi = np.mean(seed_rois)
        print(f"--> SEED {seed} AVERAGE ANNUAL ROI: {avg_roi:.2f}%")
        all_seeds_results.append(avg_roi)

    # --- FINAL REPORT ---
    print("\n========================================")
    print("      DIAGNOSTIC FINAL REPORT")
    print("========================================")
    print(f"Settings: Leverage={USE_LEVERAGE}, Ablation={ABLATION_MODE}")
    print(f"Seeds Run: {SEEDS}")
    print(f"Results (Avg ROI per seed): {all_seeds_results}")
    print(f"Mean ROI: {np.mean(all_seeds_results):.2f}%")
    print(f"Std Dev:  {np.std(all_seeds_results):.2f}")
    print("========================================")
    
    if np.std(all_seeds_results) > abs(np.mean(all_seeds_results)):
        print("FAIL: Strategy is UNSTABLE (Std Dev > Mean).")
    else:
        print("PASS: Strategy is ROBUST.")

if __name__ == "__main__":
    run_diagnostic_experiment()
# ----------------------------------------
# 5. WALK-FORWARD EXECUTION LOOP
# ----------------------------------------
# ----------------------------------------
# 5. WALK-FORWARD EXECUTION LOOP (FIXED: VALIDATION SPLIT)
# ----------------------------------------
"""