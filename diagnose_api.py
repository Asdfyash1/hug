
import requests
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    print("Checking /health...")
    try:
        t0 = time.time()
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        print(f"Time: {time.time() - t0:.2f}s")
        return True
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return False

def test_clips():
    print("\nChecking /clips (simplified)...")
    url = "https://youtu.be/uVkFrqugXFQ" # The user's URL
    try:
        t0 = time.time()
        print(f"Requesting clips for {url}...")
        resp = requests.get(f"{BASE_URL}/clips", params={
            "url": url,
            "seconds": 30,
            "num": 2
        }, timeout=60)
        print(f"Clips Status: {resp.status_code}")
        print(f"Time: {time.time() - t0:.2f}s")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Success! Got {len(data.get('clips', []))} clips")
            print(f"Video: {data.get('video_title')}")
        else:
            print(f"Error: {resp.text}")
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60s")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if test_health():
        test_clips()
