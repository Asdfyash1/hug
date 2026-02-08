import requests
import time
import sys

url = "https://kimi-agent-viral-clip-extractor.onrender.com/"
target_version = "1.3.1 (AI Enabled + Fix)"

print(f"Polling {url} for version: {target_version}")

start_time = time.time()
while time.time() - start_time < 300: # 5 minutes max
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            version = data.get("version")
            print(f"Current version: {version}")
            if version == target_version:
                print("SUCCESS: Deployment updated!")
                sys.exit(0)
        else:
            print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(10)

print("Timeout waiting for deployment.")
sys.exit(1)
