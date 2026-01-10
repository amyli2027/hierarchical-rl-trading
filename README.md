# Hierarchical Reinforcement Learning for Stock Trading
I built this project to see if an AI "Manager" could outperform individual trading strategies by dynamically switching between them. Instead of relying on just one algorithm, I created a system where a manager agent watches the market and decides how much mnoney to give to three different specialist agents (Trend Agent, Momentum Agent, and Conservative Agent) based on who is performing best.

## How it works (The Architecture)
* The Specialists: I trained three separate PPO agents using 'Stable Baselines3'. Each has a different "personality":
    * Trend Agent: Loves trends .
    * Momentum Agent: More aggressive, higher entropy (more risks).
    * Conservative Agent: Conservative, focuses on not losing money. Usually goes into treasury bonds
* The Manager: A meta-agent that looks at market conditions (like Bull vs. Bear market signals) and outputs a "voting weight" to decide how to split the portfolio between the three specialists. For example, 30% of the money goes towards investing towards one agent's decision, 30% to the other, 10% to the last one.
* The Data: I used 15 years of historical data (2009–2025) covering biotech and treasury ETFs to test volatility handling.

## How I Tested It
I have had many projects before that showed seemingly great results because of data errors, overfitting, etc so I ran many tests to ensure my program's output was realistic:

* Ablation Studies
    * I compared the manager's final ROI against each specialist running alone.
    * Goal: Prove that the total team (manager + specialists) creates a smoother equity curve than just betting it all on the aggressive "Momentum" agent.
* Overfitting Check (Walk-Forward Validation):
    * Stock data is noisy, so standard cross-validation doesn't really work.
    * I implemented a Rolling Window approach: Train on 4 years, test on the next 1 year.
    * This forces the model to constantly adapt to unseen data (like the 2020 crash or 2022 bear market) rather than memorixing history.

## Tech Stack
* Python & Pandas 
* Stable Baselines3 (PPO implementation)
* FinRL & Gymnasium 
