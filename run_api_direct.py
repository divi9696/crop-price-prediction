import uvicorn
import logging
import os

# Set detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("🚀 Starting Crop Price Prediction API...")
    print(f"📁 Current directory: {os.getcwd()}")
    print("🔧 Starting uvicorn server...")
    
    try:
        uvicorn.run(
            "src.predict_api:app",
            host="127.0.0.1",
            port=8001,
            reload=True,
            log_level="debug",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()