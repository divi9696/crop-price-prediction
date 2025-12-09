#!/bin/bash
set -e

echo "🚀 Starting AGRI PROJECT..."
echo "📁 Working directory: $(pwd)"
echo "👤 Running as: $(whoami)"
echo "🔧 Mode: $APP_MODE"

case "$APP_MODE" in
    "api")
        echo "🔌 Starting API server..."
        exec uvicorn src.predict_api:app \
            --host 0.0.0.0 \
            --port 8001 \
            --workers 1 \
            --log-level info
        ;;

    "streamlit")
        echo "🌐 Starting Streamlit app..."
        exec streamlit run app.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --server.headless true
        ;;

    "both")
        echo "🚀 Starting both API and Streamlit..."

        uvicorn src.predict_api:app \
            --host 0.0.0.0 \
            --port 8001 \
            --workers 1 \
            --log-level info &

        streamlit run app.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --server.headless true &

        wait -n
        ;;
    *)
        echo "❌ Invalid APP_MODE: $APP_MODE. Use 'api', 'streamlit', or 'both'"
        exit 1
        ;;
esac
