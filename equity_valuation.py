# ==========================================
# INDUSTRY ANALYSIS / EQUITY RESEARCH PROJECT
# Single-file Python Portfolio Project
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. Simulate Stock Price Data
# -----------------------------
days = 252
stocks = ['FintechA','BankB','TechC']

returns = np.random.multivariate_normal(
    mean=[0.0008,0.0005,0.001],
    cov=[[0.0004,0.0002,0.0001],
         [0.0002,0.0003,0.0001],
         [0.0001,0.0001,0.0005]],
    size=days
)

returns = pd.DataFrame(returns, columns=stocks)

prices = 100*(1+returns).cumprod()

# -----------------------------
# 2. Basic Risk Analysis
# -----------------------------
annual_return = returns.mean()*252
volatility = returns.std()*np.sqrt(252)
sharpe = annual_return/volatility

summary = pd.DataFrame({
    "Annual Return":annual_return,
    "Volatility":volatility,
    "Sharpe Ratio":sharpe
})

print("\nRISK METRICS")
print(summary.round(3))


# -----------------------------
# 3. Monte Carlo Price Forecast
# -----------------------------
S0 = prices['FintechA'].iloc[-1]
mu = returns['FintechA'].mean()
sigma = returns['FintechA'].std()

simulations = 1000
forecast_days = 252

paths = np.zeros((forecast_days,simulations))

for i in range(simulations):
    price=S0
    for t in range(forecast_days):
        shock=np.random.normal(mu,sigma)
        price=price*(1+shock)
        paths[t,i]=price

expected_price=paths[-1].mean()

print("\nExpected 1Y Price:",
round(expected_price,2))


# -----------------------------
# 4. Efficient Frontier
# -----------------------------
port_returns=[]
port_risk=[]
weights_list=[]

for _ in range(5000):
    weights=np.random.random(3)
    weights/=np.sum(weights)

    ret=np.sum(weights*annual_return)
    risk=np.sqrt(
        np.dot(
            weights.T,
            np.dot(returns.cov()*252,weights)
        )
    )

    port_returns.append(ret)
    port_risk.append(risk)
    weights_list.append(weights)

port_returns=np.array(port_returns)
port_risk=np.array(port_risk)

sharpe_port=port_returns/port_risk
best=np.argmax(sharpe_port)

print("\nOptimal Portfolio Weights")
for i,s in enumerate(stocks):
    print(
        s,
        round(weights_list[best][i]*100,2),
        "%"
    )


# -----------------------------
# 5. Simple DCF Valuation
# -----------------------------
fcf=50_000_000
growth=.08
discount=.12

cashflows=[]

for year in range(1,6):
    cf=fcf*((1+growth)**year)
    pv=cf/((1+discount)**year)
    cashflows.append(pv)

terminal=(fcf*(1+growth)**6)/(discount-growth)
terminal_pv=terminal/(1+discount)**5

enterprise_value=sum(cashflows)+terminal_pv

print("\nDCF Enterprise Value:")
print("${:,.0f}".format(enterprise_value))


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(14,8))

# historical prices
plt.subplot(2,2,1)
for s in stocks:
    plt.plot(prices[s],label=s)
plt.title("Simulated Stock Prices")
plt.legend()

# monte carlo
plt.subplot(2,2,2)
plt.plot(paths[:,:50],alpha=.3)
plt.title("Monte Carlo Forecast")

# efficient frontier
plt.subplot(2,2,3)
plt.scatter(
port_risk,
port_returns,
c=sharpe_port,
alpha=.5
)
plt.scatter(
port_risk[best],
port_returns[best],
marker="*",
s=300
)
plt.title("Efficient Frontier")

# risk-return
plt.subplot(2,2,4)
plt.bar(stocks,sharpe)
plt.title("Sharpe Ratios")

plt.tight_layout()
plt.show()