import pandas as pd
import matplotlib.pyplot as plt
from main import stock_trading, RobustYahooDownloader

# 1. Run the Model for the specific crash year (2022)
print("Running simulation for 2022...")
stock_trading(
    train_start_date="2009-01-01", train_end_date="2022-01-01",
    trade_start_date="2022-01-01", trade_end_date="2023-01-01",
    train_model=True, 
    model_name="final_plot_model"
)

# 2. Load the Result CSV (Created by main.py)
df = pd.read_csv("result.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 3. Normalize to $100 Start (for easy comparison)
df['AI_Strategy'] = (df['PPO'] / df['PPO'].iloc[0]) * 100
df['SPY_Benchmark'] = (df['DJI'] / df['DJI'].iloc[0]) * 100 # Note: code saves baseline as 'DJI' column name

# 4. Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['AI_Strategy'], label='Your AI Bot', color='blue', linewidth=2)
plt.plot(df.index, df['SPY_Benchmark'], label='S&P 500 (Buy & Hold)', color='gray', linestyle='--', alpha=0.7)

# Add "Crash Protection" shading
plt.fill_between(df.index, df['AI_Strategy'], df['SPY_Benchmark'], 
                 where=(df['AI_Strategy'] > df['SPY_Benchmark']), 
                 color='green', alpha=0.1, label='Alpha (Outperformance)')

plt.title('AI Strategy vs Market Crash (2022)', fontsize=16)
plt.ylabel('Portfolio Value ($100 Start)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and Show
plt.savefig("final_performance_chart.png")
print("Chart saved to 'final_performance_chart.png'")
plt.show()