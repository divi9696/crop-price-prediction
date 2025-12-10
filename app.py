"""
Enhanced Interactive Crop Price Forecast App
With advanced visualizations, real-time updates, and interactive features
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import time
import sys
import os
from typing import Dict, List, Optional, Any, Tuple

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .crop-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .price-up {
        color: #FF4B4B;
    }
    .price-down {
        color: #0066CC;
    }
    .stButton button {
        width: 100%;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .api-online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .api-offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCropForecastApp:
    def __init__(self):
        self.API_URL = "http://127.0.0.1:8001/predict"
        self.VALID_HORIZONS = [1, 2, 4, 12]  # API accepted values
        self.setup_data()
        
    def setup_data(self):
        """Initialize all data structures with enhanced crop information"""
        self.crop_data = {
            "rice": {
                "icon": "🍚",
                "varieties": ["PR126", "Basmati 1509", "Sona Masoori", "Swarna", "Pusa Basmati", "IR64", ],
                "color": "#FFD700",
                "description": "Staple food grain with seasonal price patterns influenced by monsoon and export demand",
                "season": "Kharif/Rabi",
                "growth_period": "90-150 days"
            },
            "wheat": {
                "icon": "🌾", 
                "varieties": ["Sharbati", "Lokwan", "MP Wheat", "Punjab Wheat", "HD 2967", "DBW 187"],
                "color": "#F4A460",
                "description": "Rabi crop with stable demand, influenced by government procurement policies",
                "season": "Rabi",
                "growth_period": "120-140 days"
            },
            "maize": {
                "icon": "🌽",
                "varieties": ["Hybrid", "Desi", "Sweet Corn", "Baby Corn", "Quality Protein Maize"],
                "color": "#FFA500",
                "description": "Multi-purpose crop for food, feed, and industrial use with volatile prices",
                "season": "Kharif",
                "growth_period": "90-100 days"
            },
            "pulses": {
                "icon": "🥜",
                "varieties": ["Arhar (Tur)", "Moong", "Urad", "Chana", "Masoor", "Rajma"],
                "color": "#8B4513",
                "description": "Protein-rich legumes with high price volatility and import dependency",
                "season": "Kharif/Rabi",
                "growth_period": "90-120 days"
            },
            "cotton": {
                "icon": "🧵",
                "varieties": ["BT Cotton", "Desi Cotton", "Organic Cotton", "MCU 5", "Suvin"],
                "color": "#FFFFFF",
                "description": "Cash crop with international market influence and textile industry demand",
                "season": "Kharif",
                "growth_period": "150-180 days"
            },
            "sugarcane": {
                "icon": "🎋",
                "varieties": ["Co 0238", "Co 86032", "CoM 0265", "CoJ 64", "CoS 767"],
                "color": "#FF69B4",
                "description": "Long-duration crop with government support and sugar industry linkage",
                "season": "Annual",
                "growth_period": "10-12 months"
            },
            "soybean": {
                "icon": "🫘",
                "varieties": ["JS 335", "JS 9560", "MAUS 71", "NRC 37"],
                "color": "#8B7355",
                "description": "Oilseed crop with domestic and export demand for oil and meal",
                "season": "Kharif",
                "growth_period": "90-110 days"
            }
        }
        
        self.markets_data = {
            "Delhi": {"state": "Delhi", "region": "North", "major_crops": ["wheat", "rice", "maize"]},
            "Mumbai": {"state": "Maharashtra", "region": "West", "major_crops": ["rice", "pulses", "cotton"]},
            "Kolkata": {"state": "West Bengal", "region": "East", "major_crops": ["rice", "pulses", "maize"]},
            "Chennai": {"state": "Tamil Nadu", "region": "South", "major_crops": ["rice", "cotton", "sugarcane"]},
            "Hyderabad": {"state": "Telangana", "region": "South", "major_crops": ["rice", "cotton", "maize"]},
            "Bangalore": {"state": "Karnataka", "region": "South", "major_crops": ["rice", "pulses", "soybean"]},
            "Ahmedabad": {"state": "Gujarat", "region": "West", "major_crops": ["cotton", "wheat", "pulses"]},
            "Pune": {"state": "Maharashtra", "region": "West", "major_crops": ["sugarcane", "wheat", "pulses"]},
            "Jaipur": {"state": "Rajasthan", "region": "North", "major_crops": ["wheat", "pulses", "soybean"]},
            "Lucknow": {"state": "Uttar Pradesh", "region": "North", "major_crops": ["wheat", "sugarcane", "rice"]},
            "Bhopal": {"state": "Madhya Pradesh", "region": "Central", "major_crops": ["soybean", "wheat", "pulses"]},
            "Patna": {"state": "Bihar", "region": "East", "major_crops": ["rice", "wheat", "maize"]}
        }
        
    def check_api_health(self):
        """Check if the API is available and healthy"""
        try:
            # Try the main endpoint with valid parameters
            test_payload = {
                "market": "Chennai",
                "state": "Tamil Nadu", 
                "crop": "Rice",
                "variety": "PR126",
                "horizon_weeks": 1,  # Use valid horizon
                "as_of_date": "2024-01-01"
            }
            response = requests.post(self.API_URL, json=test_payload, timeout=5)
            # If we get any response other than 404, the endpoint exists
            return response.status_code != 404
        except Exception as e:
            # Log the specific error to help with debugging (visible in Render logs)
            print(f"API Check Failed: {e}")
            return False
    
    def call_prediction_api(self, market: str, state: str, crop: str, variety: str, horizon_weeks: int, as_of_date: str) -> Tuple[Dict[str, Any], bool]:
        """Enhanced API call with proper horizon validation"""
        # Validate and format the payload with correct horizon
        payload = self.validate_and_format_payload(
            market, state, crop, variety, horizon_weeks, as_of_date
        )
        
        try:
            response = requests.post(self.API_URL, json=payload, timeout=15)
            
            if response.status_code == 200:
                return response.json(), False
            elif response.status_code == 422:
                # Validation error - get detailed error message
                error_detail = response.json().get('detail', 'Validation error')
                st.error(f"🔄 API Validation Error: {error_detail}")
                
                # Try with nearest valid horizon
                corrected_payload = self.correct_payload_for_horizon(payload, horizon_weeks)
                st.info(f"🔄 Using nearest valid horizon: {corrected_payload['horizon_weeks']} weeks")
                return self.retry_api_call(corrected_payload)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                raise Exception(f"API returned status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection Error: Cannot connect to the prediction API")
            return self.generate_enhanced_sample_data(market, state, crop, variety, horizon_weeks), True
        except requests.exceptions.Timeout:
            st.warning("⏰ API Timeout: Using simulated data")
            return self.generate_enhanced_sample_data(market, state, crop, variety, horizon_weeks), True
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
            return self.generate_enhanced_sample_data(market, state, crop, variety, horizon_weeks), True
    
    def validate_and_format_payload(self, market: str, state: str, crop: str, variety: str, horizon_weeks: int, as_of_date: str) -> Dict[str, Any]:
        """Validate and format the API payload to avoid 422 errors"""
        
        # Convert date to string if it's a date object
        if isinstance(as_of_date, datetime):
            as_of_date_str = as_of_date.strftime("%Y-%m-%d")
        else:
            as_of_date_str = str(as_of_date)
        
        # Ensure horizon_weeks is one of the valid values
        valid_horizon = self.get_nearest_valid_horizon(horizon_weeks)
        
        # Clean and validate string inputs
        market_clean = market.strip()
        state_clean = state.strip()
        crop_clean = crop.strip().lower()
        variety_clean = variety.strip()
        
        payload = {
            "market": market_clean,
            "state": state_clean,
            "crop": crop_clean,
            "variety": variety_clean,
            "horizon_weeks": valid_horizon,
            "as_of_date": as_of_date_str
        }
        
        return payload
    
    def get_nearest_valid_horizon(self, requested_horizon: int) -> int:
        """Get the nearest valid horizon from the allowed values"""
        # Find the closest valid horizon
        closest = min(self.VALID_HORIZONS, key=lambda x: abs(x - requested_horizon))
        return closest
    
    def correct_payload_for_horizon(self, original_payload: Dict, requested_horizon: int) -> Dict:
        """Correct payload specifically for horizon issues"""
        corrected = original_payload.copy()
        
        # Use the nearest valid horizon
        corrected["horizon_weeks"] = self.get_nearest_valid_horizon(requested_horizon)
        
        return corrected
    
    def retry_api_call(self, payload: Dict) -> Tuple[Dict[str, Any], bool]:
        """Retry API call with corrected payload"""
        try:
            response = requests.post(self.API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json(), False
            else:
                raise Exception(f"Retry failed with status: {response.status_code}")
        except:
            # Generate sample data for the original request parameters
            return self.generate_enhanced_sample_data(
                payload["market"], payload["state"], payload["crop"], 
                payload["variety"], payload["horizon_weeks"]
            ), True
    
    def generate_enhanced_sample_data(self, market: str, state: str, crop: str, variety: str, horizon_weeks: int) -> Dict[str, Any]:
        """Generate more realistic sample data with market-specific patterns"""
        # Set seed for reproducible results
        seed_value = hash(market + state + crop + variety) % 10000
        np.random.seed(seed_value)
        
        # Base prices with more realistic variations
        base_prices = {
            "rice": 2200, "wheat": 1800, "maize": 1600, "pulses": 3500, 
            "cotton": 2800, "sugarcane": 1900, "soybean": 3200
        }
        
        # Market-specific multipliers with regional variations
        market_multipliers = {
            "Delhi": 1.15, "Mumbai": 1.1, "Kolkata": 1.0, "Chennai": 0.95,
            "Hyderabad": 0.98, "Bangalore": 1.05, "Ahmedabad": 0.92, 
            "Pune": 0.96, "Jaipur": 0.88, "Lucknow": 0.85, "Bhopal": 0.90, "Patna": 0.87
        }
        
        # Crop-specific volatility factors
        volatility_factors = {
            "rice": 0.08, "wheat": 0.06, "maize": 0.12, "pulses": 0.15,
            "cotton": 0.18, "sugarcane": 0.04, "soybean": 0.14
        }
        
        base_price = base_prices.get(crop, 2000) * market_multipliers.get(market, 1.0)
        volatility = volatility_factors.get(crop, 0.1)
        
        predictions = []
        current_date = datetime.now()
        
        for i in range(horizon_weeks):
            date = (current_date + timedelta(weeks=i+1)).strftime("%Y-%m-%d")
            
            # Enhanced price modeling with trend, seasonality, and noise
            trend = base_price * (1 + (i * 0.012))  # 1.2% weekly trend
            seasonal = np.sin(i * 0.5) * base_price * 0.03  # 3% seasonal variation
            cycle = np.cos(i * 0.3) * base_price * 0.02  # Cyclical patterns
            noise = np.random.normal(0, base_price * volatility * 0.3)  # Random noise
            
            modal_price = max(base_price * 0.5, trend + seasonal + cycle + noise)
            
            # Calculate realistic min/max prices
            price_range = modal_price * volatility
            min_price = max(modal_price * 0.7, modal_price - price_range * 0.8)
            max_price = modal_price + price_range * 0.8
            
            predictions.append({
                "date": date,
                "modal_price": round(modal_price, 2),
                "min_price": round(min_price, 2),
                "max_price": round(max_price, 2),
                "week": i + 1
            })
        
        # Enhanced model metrics
        return {
            "predictions": predictions,
            "model_version": "advanced-model-2.1",
            "market_analysis": {
                "trend": "bullish" if predictions[-1]['modal_price'] > predictions[0]['modal_price'] else "bearish",
                "volatility_category": "high" if volatility > 0.12 else "medium" if volatility > 0.08 else "low",
                "market_efficiency": round(0.75 + np.random.random() * 0.2, 2)
            },
            "metrics": {
                "rmse": round(np.random.uniform(25, 55), 2),
                "mae": round(np.random.uniform(20, 45), 2),
                "mape": round(np.random.uniform(1.0, 2.5), 2),
                "r2": round(np.random.uniform(0.85, 0.96), 3),
                "confidence": round(0.82 + np.random.random() * 0.15, 2),
                "volatility": round(volatility * 100, 1)
            },
            "api_status": "simulated_data",
            "requested_horizon": horizon_weeks,
            "actual_horizon": horizon_weeks
        }

    def create_comprehensive_forecast_chart(self, predictions: List[Dict], crop: str, market: str, requested_horizon: int, actual_horizon: int) -> go.Figure:
        """Create enhanced interactive forecast visualization"""
        df = pd.DataFrame(predictions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Adjust title if horizon was modified
        if requested_horizon != actual_horizon:
            title_note = f" (Adjusted from {requested_horizon} to {actual_horizon} weeks)"
        else:
            title_note = ""
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'📈 {crop.title()} Price Forecast - {market}{title_note}',
                '📊 Weekly Price Changes (%)',
                '🎯 Price Range Analysis'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        crop_color = self.crop_data[crop]['color']
        
        # Main price line
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['modal_price'],
                mode='lines+markers+text',
                name='Modal Price',
                line=dict(color=crop_color, width=4),
                marker=dict(size=8, symbol='circle'),
                text=[f'₹{x:,.0f}' for x in df['modal_price']],
                textposition="top center",
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Price: ₹%{y:,.2f}<extra></extra>'
            ), row=1, col=1
        )
        
        # Confidence interval area
        if 'min_price' in df.columns and 'max_price' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'].tolist() + df['date'].tolist()[::-1],
                    y=df['max_price'].tolist() + df['min_price'].tolist()[::-1],
                    fill='toself',
                    fillcolor=f'rgba{(*self.hex_to_rgb(crop_color), 0.2)}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Expected Range',
                    showlegend=True,
                    hovertemplate='Price Range: ₹%{y:,.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # Weekly price changes
        df['price_change_pct'] = df['modal_price'].pct_change() * 100
        df['price_change_pct'] = df['price_change_pct'].fillna(0)
        
        fig.add_trace(
            go.Bar(
                x=df['date'], y=df['price_change_pct'],
                name='Weekly Change %',
                marker_color=['red' if x < 0 else 'green' for x in df['price_change_pct']],
                hovertemplate='<b>%{x|%d %b}</b><br>Change: %{y:.2f}%<extra></extra>'
            ), row=2, col=1
        )
        
        # Price range analysis (max-min spread)
        if 'min_price' in df.columns and 'max_price' in df.columns:
            df['price_spread'] = df['max_price'] - df['min_price']
            df['spread_pct'] = (df['price_spread'] / df['modal_price']) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['spread_pct'],
                    mode='lines+markers',
                    name='Price Range %',
                    line=dict(color='purple', width=3, dash='dot'),
                    marker=dict(size=6),
                    hovertemplate='<b>%{x|%d %b}</b><br>Range: %{y:.1f}% of price<extra></extra>'
                ), row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Change (%)", row=2, col=1)
        fig.update_yaxes(title_text="Range (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def render_enhanced_sidebar(self):
        """Render the interactive sidebar with enhanced features"""
        with st.sidebar:
            st.markdown('<div class="main-header">🌾 Crop Intelligence</div>', unsafe_allow_html=True)
            
            # API Status Indicator
            api_online = self.check_api_health()
            status_class = "api-online" if api_online else "api-offline"
            status_text = "🟢 API Online" if api_online else "🔴 API Offline"
            
            st.markdown(f'<div class="api-status {status_class}">{status_text}</div>', unsafe_allow_html=True)
            
            if not api_online:
                st.info("""
                💡 **Working in Simulation Mode**
                - Using enhanced simulated data
                - All features available
                - Real data when API connects
                """)
            
            # Market selection with enhanced info
            st.subheader("📍 Market Selection")
            selected_market = st.selectbox(
                "Choose Market Center",
                options=list(self.markets_data.keys()),
                format_func=lambda x: f"{x} ({self.markets_data[x]['region']})",
                help="Select agricultural market for price analysis",
                index=3
            )
            
            # Display market information
            market_info = self.markets_data[selected_market]
            st.info(f"""
            **Market Details:**
            - **State:** {market_info['state']}
            - **Region:** {market_info['region']}
            - **Major Crops:** {', '.join(market_info['major_crops'])}
            """)
            
            # Crop selection with enhanced display
            st.subheader("🌱 Crop Selection")
            selected_crop = st.selectbox(
                "Select Crop",
                options=list(self.crop_data.keys()),
                format_func=lambda x: f"{self.crop_data[x]['icon']} {x.title()}",
                help="Choose crop for price forecasting"
            )
            
            # Display crop information
            crop_info = self.crop_data[selected_crop]
            with st.expander("📖 Crop Information", expanded=True):
                st.write(f"**Description:** {crop_info['description']}")
                st.write(f"**Season:** {crop_info['season']}")
                st.write(f"**Growth Period:** {crop_info['growth_period']}")
            
            # Variety selection
            selected_variety = st.selectbox(
                "Select Variety/Quality",
                options=crop_info['varieties'],
                help="Choose specific variety or quality grade"
            )
            
            # Enhanced forecast parameters
            st.subheader("⚙️ Forecast Parameters")
            
            # Horizon selection with API constraints
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write("**📅 Forecast Horizon**")
            st.write("API supports: 1, 2, 4, or 12 weeks")
            
            # Use selectbox instead of slider for valid values
            horizon_options = {1: "1 week", 2: "2 weeks", 4: "4 weeks", 12: "12 weeks"}
            selected_horizon = st.selectbox(
                "Select Forecast Horizon",
                options=list(horizon_options.keys()),
                format_func=lambda x: horizon_options[x],
                index=2,  # Default to 4 weeks
                help="Choose from API-supported horizon values"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            as_of_date = st.date_input(
                "Analysis Reference Date",
                value=datetime.now(),
                help="Base date for forecast calculations"
            )
            
            # Action buttons
            st.subheader("🚀 Actions")
            col1, col2 = st.columns(2)
            with col1:
                forecast_btn = st.button(
                    "Generate Forecast",
                    use_container_width=True,
                    type="primary"
                )
            with col2:
                if st.button("Clear Analysis", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            
            # Quick analysis options
            st.subheader("🔍 Quick Analysis")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Multi-Crop Compare", use_container_width=True):
                    self.run_multi_crop_analysis()
            with col2:
                if st.button("Market Trends", use_container_width=True):
                    self.run_market_trend_analysis()
            
            return (forecast_btn, selected_market, market_info['state'], 
                   selected_crop, selected_variety, selected_horizon, as_of_date, api_online)

    def render_advanced_metrics(self, data: Dict, crop: str, market: str, using_sample: bool):
        """Render enhanced metrics dashboard"""
        if not data.get('predictions'):
            return
            
        predictions = data['predictions']
        metrics = data.get('metrics', {})
        market_analysis = data.get('market_analysis', {})
        
        # Data source indicator
        if using_sample:
            st.warning("📊 Displaying Enhanced Simulation Data")
        else:
            st.success("📊 Live API Data")
        
        # Show horizon info if it was adjusted
        requested_horizon = data.get('requested_horizon')
        actual_horizon = data.get('actual_horizon', len(predictions))
        if requested_horizon and requested_horizon != actual_horizon:
            st.info(f"🔧 Horizon adjusted from {requested_horizon} to {actual_horizon} weeks for API compatibility")
        
        # Enhanced metrics cards
        st.subheader("📊 Market Intelligence Dashboard")
        
        # Row 1: Primary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = predictions[0]['modal_price']
            final_price = predictions[-1]['modal_price']
            total_change = ((final_price - current_price) / current_price) * 100
            
            st.metric(
                label="Current Price",
                value=f"₹{current_price:,.0f}",
                delta=f"{total_change:+.1f}% overall",
                delta_color="inverse"
            )
        
        with col2:
            avg_price = np.mean([p['modal_price'] for p in predictions])
            price_std = np.std([p['modal_price'] for p in predictions])
            
            st.metric(
                label="Forecast Average ± Volatility",
                value=f"₹{avg_price:,.0f}",
                delta=f"±₹{price_std:,.0f}",
            )
        
        with col3:
            max_week = np.argmax([p['modal_price'] for p in predictions]) + 1
            min_week = np.argmin([p['modal_price'] for p in predictions]) + 1
            
            st.metric(
                label="Peak/Trough Timing",
                value=f"Week {max_week}",
                delta=f"Week {min_week}",
            )
        
        with col4:
            confidence = metrics.get('confidence', 0) * 100
            model_quality = "High" if confidence > 85 else "Medium" if confidence > 75 else "Low"
            
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.1f}%",
                delta=model_quality,
            )

    def run(self):
        """Main application runner"""
        # Initialize session state
        if 'forecast_data' not in st.session_state:
            st.session_state.forecast_data = None
        if 'using_sample' not in st.session_state:
            st.session_state.using_sample = False
        if 'last_analysis' not in st.session_state:
            st.session_state.last_analysis = None
        
        # Render enhanced sidebar
        (forecast_btn, selected_market, state, selected_crop, 
         selected_variety, horizon_weeks, as_of_date, api_online) = self.render_enhanced_sidebar()
        
        # Handle forecast generation
        if forecast_btn:
            with st.spinner("🔮 Generating advanced crop intelligence forecast..."):
                # Enhanced progress visualization
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔄 Initializing market analysis...")
                    elif i < 60:
                        status_text.text("📊 Processing historical patterns...")
                    elif i < 90:
                        status_text.text("🤖 Applying AI forecasting models...")
                    else:
                        status_text.text("✅ Finalizing insights...")
                
                # API call with enhanced parameters
                data, is_sample = self.call_prediction_api(
                    selected_market, state, selected_crop, 
                    selected_variety, horizon_weeks, as_of_date
                )
                
                # Store in session state
                st.session_state.forecast_data = data
                st.session_state.using_sample = is_sample
                st.session_state.last_analysis = {
                    'crop': selected_crop,
                    'market': selected_market,
                    'variety': selected_variety,
                    'horizon': horizon_weeks,
                    'date': as_of_date
                }
                
                progress_bar.empty()
                status_text.empty()
        
        # Display results if available
        if st.session_state.forecast_data:
            data = st.session_state.forecast_data
            last_analysis = st.session_state.last_analysis
            using_sample = st.session_state.using_sample
            
            # Get horizon information
            requested_horizon = data.get('requested_horizon', last_analysis['horizon'])
            actual_horizon = data.get('actual_horizon', len(data['predictions']))
            
            # Enhanced header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h1 style="margin: 0; font-size: 2.8rem; text-align: center;">
                    {self.crop_data[last_analysis['crop']]['icon']} 
                    {last_analysis['crop'].title()} Intelligence Report
                </h1>
                <p style="margin: 0.5rem 0; font-size: 1.3rem; text-align: center; opacity: 0.9;">
                    {last_analysis['market']} • {last_analysis['variety']} • {actual_horizon} weeks
                </p>
                <p style="margin: 0; font-size: 1rem; text-align: center; opacity: 0.7;">
                    Generated on {datetime.now().strftime('%d %b %Y, %H:%M')} • {data.get('model_version', 'Advanced Model')}
                    {' • 🧪 Simulation Mode' if using_sample else ' • 🟢 Live Data'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Render dashboard components
            self.render_advanced_metrics(data, last_analysis['crop'], last_analysis['market'], using_sample)
            
            # Enhanced visualization
            st.plotly_chart(
                self.create_comprehensive_forecast_chart(
                    data['predictions'], last_analysis['crop'], last_analysis['market'],
                    requested_horizon, actual_horizon
                ), 
                use_container_width=True
            )
            
        else:
            # Enhanced welcome state
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        border-radius: 15px; margin: 2rem 0;">
                <h1 style="font-size: 3.5rem; color: #2E8B57; margin-bottom: 1.5rem;">🌾 Advanced Crop Intelligence</h1>
                <p style="font-size: 1.4rem; color: #555; margin-bottom: 3rem; line-height: 1.6;">
                    AI-powered agricultural price forecasting platform<br>
                    Make data-driven decisions with confidence
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature highlights
            st.subheader("🚀 Platform Capabilities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("""
                **📈 Advanced Forecasting**
                - Multi-week price predictions
                - Confidence intervals
                - Trend and pattern analysis
                - Volatility assessment
                """)
            
            with col2:
                st.info("""
                **🌾 Comprehensive Coverage**
                - 7 major crops with varieties
                - 12 market centers across India
                - Regional price patterns
                - Seasonal adjustments
                """)
            
            with col3:
                st.info("""
                **💡 Intelligent Insights**
                - AI-powered recommendations
                - Risk assessment
                - Trading strategies
                - Market efficiency scores
                """)

# Run the application
if __name__ == "__main__":
    app = AdvancedCropForecastApp()
    app.run()