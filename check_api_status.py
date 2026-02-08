import requests
import sys

url = "https://kimi-agent-viral-clip-extractor.onrender.com/clips?url=https://youtu.be/uVkFrqugXFQ&mode=nvidia&nvidia_key=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&num=3"

try:
    print(f"Checking URL: {url}")
    response = requests.get(url, timeout=30)
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(response.json())
    except:
        print("Response Text:")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
