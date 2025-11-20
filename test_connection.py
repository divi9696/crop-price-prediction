import requests

try:
    response = requests.get("http://127.0.0.1:8001/health", timeout=5)
    print(f"✅ API Connection SUCCESS: {response.status_code}")
    print(response.json())
except Exception as e:
    print(f"❌ API Connection FAILED: {e}")