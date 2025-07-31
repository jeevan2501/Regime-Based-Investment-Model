# Regime-Based Investment Strategy Model 🧠📈

This project uses time-series data and unsupervised machine learning (KMeans Clustering) to classify the market into **Bull**, **Bear**, and **Volatile** regimes using the S&P 500 Index.

## 📌 Objective
To help investors dynamically shift their asset allocation based on current market conditions using a regime-detection model.

## 🔍 How It Works
- Data Source: Yahoo Finance (S&P 500)
- Features: Returns, Volatility, Momentum
- Model: KMeans Clustering (3 clusters)
- Output: Regime-labeled S&P 500 Chart

## 📈 Investment Strategy
- **Bull** → Invest in equity, index funds, crypto
- **Bear** → Shift to gold, bonds, defensive stocks
- **Volatile** → Stay partially in cash or hedge

## 📊 Tech Stack
- Python
- pandas, numpy, matplotlib, seaborn
- sklearn (KMeans)
- yfinance for data extraction
- Jupyter Notebook

## ✅ Current Recommended Regime
As per the latest model output: **Bull Market**

## 📂 How to Run
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Open notebook: `jupyter notebook Regime_Based_Model.ipynb`

---

## 🤝 Connect
Made by **Jeevan Jose**  
📍 Trivandrum | 💼 BCom Graduate | 📊 Aspiring Data & Investment Analyst  
