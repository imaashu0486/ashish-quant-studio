# Ashish Quant Studio

Ashish Quant Studio is a Django-based quantitative finance application that combines
machine learning, technical analysis, and live market data to analyze and predict
NSE stocks and indices.  
The project is built to demonstrate a complete end-to-end ML system rather than
isolated notebooks or scripts.

---

## Problem Statement

Retail market analysis often relies on individual indicators or subjective judgement.
This project explores whether combining multiple technical indicators with machine
learning models can provide more structured, data-driven insights for short-term
market analysis.

---

## Key Features

- Stock price prediction using regression models
- Index direction prediction using classification models
- Feature engineering based on technical indicators
- Interactive data visualization using Plotly
- Market overview with indices and movers
- Clean Django-based web interface

---

## Machine Learning Models Used

- Ridge Regression  
- Random Forest (Regressor and Classifier)  
- Logistic Regression  
- XGBoost  
- Simple ensemble using averaged predictions  

Models are trained on historical market data using time-aware feature engineering
to reduce data leakage.

---

## Technical Indicators

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Exponential and Simple Moving Averages (EMA, SMA)
- Bollinger Bands
- Average True Range (ATR)
- Average Directional Index (ADX)
- Stochastic Oscillator
- Volume-based indicators

---

## Tech Stack

- Backend: Django
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost
- Market Data: Yahoo Finance (yfinance)
- Visualization: Plotly
- Frontend Styling: Tailwind CSS

---

## System Architecture (High Level)

Market Data  
→ Feature Engineering  
→ Model Training & Evaluation  
→ Prediction Generation  
→ Web-Based Visualization  

The application follows a modular structure to keep data collection,
ML logic, and web views decoupled.

---

## Local Setup & Execution

```bash
git clone https://github.com/imaashu0486/ashish-quant-studio.git
cd ashish-quant-studio
python -m venv venv
```
Activate the virtual environment:
```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```
Install dependencies and start the server:
```bash
pip install -r requirements.txt
python manage.py runserver
```

Access the application at:
```bash
http://127.0.0.1:8000/
```


## Deployment

The application is designed to be deployment-ready on servers with
sufficient CPU and memory resources.

During testing, free-tier cloud platforms were found to impose strict
memory and disk limits that are not suitable for ML-heavy workloads
involving pandas, XGBoost, and live market data fetching.

For this reason, the project is demonstrated locally for evaluation.
The same codebase can be deployed without modification on
higher-resource or paid cloud environments.

---

## Infrastructure Considerations

- ML libraries such as pandas and XGBoost have high memory footprints
- Live market data fetching increases startup latency
- Concurrent requests amplify resource usage
- Free-tier platforms aggressively terminate long-running processes

These constraints influenced the deployment strategy for this project.

---

## Security Notes

- The project is intended for educational use
- No user authentication or financial transactions are implemented
- No trading actions are executed
- All predictions are informational only

---

## Limitations

- Market predictions are probabilistic and not guaranteed
- Yahoo Finance data may be delayed or unavailable at times
- This project is not intended for real trading decisions

---

## Future Improvements

- Caching market data to reduce repeated API calls
- Asynchronous data fetching
- Model retraining pipelines
- Containerized deployment using Docker
- Scalable cloud deployment with autoscaling

---

## License

This project is licensed under the MIT License.

---

## Author

**Ashish Ranjan**  
Interests: Quantitative Finance, Machine Learning, Backend Development

