# %%
import yfinance as yf

tickers = ['SETFNIF50.NS', 'GOLDBEES.NS']
data = yf.download(tickers, start="2015-01-01")

print(data.columns)


# %%
# Extract only the 'Close' prices
close = data.loc[:, ('Close', slice(None))]
close.columns = close.columns.droplevel(0)  # Flatten column names

# View output
print(close.head())


# %%
returns = close.pct_change().dropna()
returns.head()

# %%


# %%
from hmmlearn.hmm import GaussianHMM
import numpy as np
import matplotlib.pyplot as plt

# Convert Nifty ETF returns to numpy array for HMM
nifty_returns = returns['SETFNIF50.NS'].values.reshape(-1, 1)

# Fit HMM model with 3 regimes
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(nifty_returns)

# Predict hidden states (regimes)
hidden_states = model.predict(nifty_returns)

# Add to DataFrame for visualization
returns['Regime'] = hidden_states


# %%
print(returns['Regime'].value_counts())


# %%
aligned_close = data['Close']['SETFNIF50.NS'].loc[returns.index]


# %%
plt.figure(figsize=(14, 6))

aligned_close = data['Close']['SETFNIF50.NS'].loc[returns.index]  # align dates

for i in range(3):
    state = returns['Regime'] == i
    plt.plot(aligned_close.index[state], aligned_close[state], '.', label=f'Regime {i}')

plt.legend()
plt.title("Market Regimes Detected on NIFTY ETF")
plt.show()


# %%
# Calculate average return per regime
returns['Daily Return'] = aligned_close.pct_change()
regime_stats = returns.groupby('Regime')['Daily Return'].mean()
print(regime_stats)


# %%
regime_map = {
    0: 'Bull',
    1: 'Sideways',
    2: 'Bear'
}
returns['Regime_Label'] = returns['Regime'].map(regime_map)


# %%
def signal(row):
    if row['Regime_Label'] == 'Bull':
        return 1   # Invest
    elif row['Regime_Label'] == 'Bear':
        return -1  # Avoid or Short
    else:
        return 0   # Stay neutral

returns['Signal'] = returns.apply(signal, axis=1)


# %%
returns['Strategy_Return'] = returns['Daily Return'] * returns['Signal']
returns[['Daily Return', 'Strategy_Return']].cumsum().plot(figsize=(14,6))
plt.title('Buy & Hold vs Regime-Based Strategy')
plt.grid()
plt.show()


# %%
import pandas as pd

# Example data
data = {
    'regime': ['bull', 'bear', 'neutral', 'bull', 'bear']
}

# Create the DataFrame
returns_df = pd.DataFrame(data)


# %%
def get_allocation(regime):
    if regime == 'bull':
        return [0.8, 0.1, 0.1]  # Nifty, Gold, Cash
    elif regime == 'bear':
        return [0.3, 0.5, 0.2]
    else:
        return [0.5, 0.3, 0.2]

# Apply the allocation function
allocations = returns_df['regime'].apply(get_allocation)

# Convert to DataFrame
allocations_df = pd.DataFrame(
    allocations.tolist(),
    index=returns_df.index,
    columns=['Nifty', 'Gold', 'Cash']
)

allocations_df.head()


# %%
# Assuming you have this from earlier steps (returns per asset)
# Make sure the index matches `allocations_df`
# Example:
# returns = pd.DataFrame({ 'Nifty': ..., 'Gold': ..., 'Cash': ... }, index=returns_df.index)

portfolio_returns = (returns * allocations_df).sum(axis=1)
portfolio_returns.head()


# %%
cumulative_returns = (1 + portfolio_returns).cumprod()
cumulative_returns.plot(figsize=(12, 6), title='Cumulative Portfolio Returns')


# %%
print(returns.columns)


# %%
returns = returns.rename(columns={'SETFNIF50.NS': 'Nifty'})


# %%
nifty_cum_returns = (1 + returns['Nifty']).cumprod()
cumulative_returns.plot(label='Regime-based Portfolio')
nifty_cum_returns.plot(label='Nifty Only')
plt.legend()
plt.title('Strategy vs Nifty Benchmark')


# %%
print(returns.columns)


# %%
allocations_df = pd.DataFrame(
    allocations.tolist(),
    index=returns_df.index,
    columns=['Nifty', 'Gold', 'Cash']
)


# %%
(allocations_df.add(1).cumprod()).plot(figsize=(12,6))
plt.title('Cumulative Portfolio Growth by Regime Allocation')
plt.ylabel('Growth Factor')
plt.xlabel('Time')
plt.grid(True)
plt.show()




# %%
