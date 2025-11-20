🌾 Crop Price Prediction System
A comprehensive end-to-end machine learning system for predicting short-term and medium-term crop prices in Indian markets. This project provides accurate price forecasts for major crops like rice, wheat, maize, pulses, cotton, and sugarcane.

🚀 Features
Multi-Crop Support: Predict prices for rice, wheat, maize, pulses, cotton, sugarcane

Multiple Forecasting Horizons: 1, 2, 4, 8, and 12-week predictions

Advanced ML Models: LightGBM, LSTM, and ensemble methods

Production-Ready API: FastAPI with automatic documentation

Interactive Web Interface: Streamlit dashboard with real-time visualizations

Market Coverage: 10 major Indian markets with state-wise analysis

Data Export: Download forecasts as CSV or JSON


Python 3.9 or higher

pip (Python package manager)

Git

Step 1: Clone and Setup Environment
bash
# Clone the repository
git clone <repository-url>
cd crop-price-prediction

# Create virtual environment
python -m venv crop_env

# Activate environment
# On Windows:
crop_env\Scripts\activate
# On macOS/Linux:
source crop_env/bin/activate
Step 2: Install Dependencies
bash
pip install -r requirements.txt
Step 3: Generate Synthetic Data
bash
python data/generate_synthetic.py
This creates realistic crop price data for:

10 Indian markets: Kolkata, Delhi, Mumbai, Chennai, Hyderabad, Bangalore, Ahmedabad, Pune, Jaipur, Lucknow

3 major crops: Rice, Wheat, Maize

3+ years of daily data with seasonal patterns

Step 4: Train the Models
bash
# Train LightGBM model (recommended)
python src/train.py --config config.yml --data data/synthetic_crop_prices.csv --model lightgbm

# Train LSTM model
python src/train.py --model lstm

# Train both models
python src/train.py --model both

# Train for specific crop
python src/train.py --crop wheat
🎯 Usage
Option 1: API Mode (Recommended)
Start the API Server
bash
uvicorn src.predict_api:app --host 127.0.0.1 --port 8001 --reload
API Endpoints
Health Check: GET http://127.0.0.1:8001/health

Make Prediction: POST http://127.0.0.1:8001/predict

Interactive Docs: http://127.0.0.1:8001/docs

Example API Request
bash
curl -X POST "http://127.0.0.1:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "market": "Kolkata",
       "state": "West Bengal",
       "crop": "rice",
       "variety": "PR 126",
       "horizon_weeks": 4,
       "as_of_date": "2024-10-01"
     }'
Example API Response
json
{
  "predictions": [
    {
      "date": "2024-10-08",
      "modal_price": 2300.5,
      "min_price": 2200.0,
      "max_price": 2400.0,
      "confidence_interval": {"lower": 2185.48, "upper": 2415.53}
    }
  ],
  "model_version": "v1.0",
  "metrics": {
    "rmse": 45.3,
    "mae": 36.7,
    "mape": 1.5,
    "r2": 0.89
  }
}
Option 2: Web Interface
Start Streamlit App
bash
streamlit run app.py
Then open http://localhost:8501 in your browser.

Web Interface Features
Input Form: Select market, state, crop, variety, and forecast horizon

Interactive Charts: Plotly visualizations with price ranges

Data Export: Download forecasts as CSV or copy JSON

Real-time Metrics: Model performance indicators

Sample Data: Test with pre-configured examples

📊 Model Performance
Supported Crops & Varieties
Rice: PR 126, Basmati, Sona Masoori, Swarna, Pusa Basmati, IR64

Wheat: Sharbati, Lokwan, MP Wheat, Punjab Wheat, HD 2967, DBW 17

Maize: Hybrid, Desi, Sweet Corn, Baby Corn, Quality Protein Maize

Pulses: Arhar, Moong, Urad, Chana, Masoor, Rajma

Cotton: BT Cotton, Desi Cotton, Organic Cotton, ELS Cotton

Sugarcane: Co 0238, Co 86032, CoM 0265, CoC 671

Forecast Horizons
1 week (short-term)

2 weeks (short-term)

4 weeks (medium-term)

8 weeks (medium-term)

12 weeks (long-term)

🔧 Configuration
Edit config.yml to customize:

yaml
model:
  target: "modal_price"           # Target variable
  horizons: [1, 2, 4, 12]         # Forecast horizons
  aggregation: "weekly"           # Data frequency
  target_crop: "rice"             # Default crop

features:
  lags: [1, 2, 4, 12]            # Lag features
  rolling_windows: [4, 12]        # Rolling statistics

lightgbm:
  n_estimators: 1000             # Number of trees
  learning_rate: 0.1             # Learning rate
  max_depth: 8                   # Tree depth
🐳 Docker Deployment
Build and Run with Docker
bash
cd docker
docker-compose up --build
Docker Services
API Server: http://localhost:8000

Redis Cache: localhost:6379 (optional)

📈 Model Architecture
Feature Engineering
Time Features: Year, month, quarter, week, seasonal flags

Lag Features: 1, 2, 4, 12-period lags

Rolling Statistics: 4 and 12-period moving averages

Categorical Encoding: Target encoding for markets/states

External Factors: Weather, arrivals, policy flags

Machine Learning Models
LightGBM: Gradient boosting with time-series cross-validation

LSTM: Deep learning for sequence prediction

Multi-horizon: Separate models for each forecast period

Evaluation Metrics
RMSE (Root Mean Square Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

R² (Coefficient of Determination)

🔍 Monitoring & Debugging
API Health Check
bash
curl http://localhost:8001/health
Streamlit Debug Mode
Expand "Developer Options" in the web interface

Test direct imports and HTTP connections

View environment information and logs

Common Issues
Port conflicts: Use different ports (8002, 8003, etc.)

Import errors: Ensure you're in the project root directory

Model loading: Verify model files exist in models/ directory

Data issues: Regenerate synthetic data if needed

📚 Data Sources
Synthetic Data (Development)
Realistic price patterns with seasonality

Market-specific variations

Weather and arrival simulations

Government policy effects

Real Data Integration
When ready for production, integrate with:

AGMARKNET: Government mandi price data

IMD: Indian Meteorological Department

State Agriculture Portals: Regional crop data

FAOSTAT: International agriculture statistics

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🆘 Support
For issues and questions:

Check the debug information in the Streamlit app

Verify API is running on correct port

Ensure all dependencies are installed

Check the model files are properly generated

🎯 Quick Start Recap
bash
# 1. Setup
python -m venv crop_env && crop_env\Scripts\activate
pip install -r requirements.txt

# 2. Data & Models
python data/generate_synthetic.py
python src/train.py

# 3. Run Services
uvicorn src.predict_api:app --host 127.0.0.1 --port 8001 --reload
streamlit run app.py

# 4. Access
# API: http://127.0.0.1:8001/docs
# Web: http://localhost:8501


Happy Forecasting! 🌾📈

