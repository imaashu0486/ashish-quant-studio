# 📈 Ashish Quant Studio
An End-to-End Quantitative Finance & Machine Learning Dashboard

Ashish Quant Studio is a full-stack Django-based quantitative finance platform that applies machine learning, technical analysis, and real-time market data to analyze NSE stocks and indices. This project demonstrates a complete ML workflow — from data ingestion to model deployment.

---

## 🚀 Key Features
- 📊 Stock price prediction using regression models  
- 📉 Index direction prediction using classification models  
- 🔥 NIFTY 50 heatmap and daily market movers  
- ⏱ Intraday candlestick charts with Plotly  
- 🤖 Multiple ML models with ensemble learning  
- 🌗 Dark / Light theme support  
- 📁 Auto-generated CSV files for training and testing  

---

## 🧠 Machine Learning Models
- Ridge Regression  
- Random Forest (Regressor & Classifier)  
- Logistic Regression  
- XGBoost (Regressor & Classifier)  
- Ensemble (Average of best-performing models)  

---

## 📐 Technical Indicators Used
- RSI (Relative Strength Index)  
- MACD (Moving Average Convergence Divergence)  
- EMA & SMA  
- Bollinger Bands  
- ATR (Average True Range)  
- ADX (Average Directional Index)  
- Stochastic Oscillator  
- OBV (On-Balance Volume)  

All indicators are engineered in a leak-free manner using historical data only.

---

## 🛠 Tech Stack
- **Backend:** Django  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Market Data:** Yahoo Finance (yfinance)  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Plotly  
- **UI:** Tailwind CSS  
- **Deployment:** Render / Railway  
- **Database:** SQLite (demo purpose)  

---

## 🖥 Application Capabilities
- Search NSE stocks and indices  
- Next-day stock price prediction  
- Probability-based index direction prediction  
- Interactive and responsive charts  
- Downloadable CSV outputs for analysis  
- Robust data fetching with fallback handling  

---

## ⚙️ Local Setup Instructions

```bash
git clone https://github.com/imaashu0486/ashish-quant-studio.git
cd ashish-quant-studio
pip install -r requirements.txt
python manage.py runserver
