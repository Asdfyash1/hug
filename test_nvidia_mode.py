
import requests
import os
import json
import sys

# REPLACE THIS WITH YOUR NVIDIA KEY or set NVIDIA_API_KEY env var
# Get from: https://build.nvidia.com/explore/discover
DEFAULT_KEY = "YOUR_NVIDIA_API_KEY_HERE" 
API_KEY = os.environ.get("NVIDIA_API_KEY", DEFAULT_KEY)

URL = "https://youtu.be/uVkFrqugXFQ"
BASE_URL = "http://127.0.0.1:5000"

def test_nvidia():
    print("Testing Nvidia/DeepSeek Mode...")
    
    # Check if key is set (simple length check or placeholder check)
    if "YOUR_NVIDIA" in API_KEY:
        print("\nWARNING: You haven't set your Nvidia API Key!")
        print("1. Get a key here: https://build.nvidia.com/explore/discover")
        print("2. Open this script and replace 'YOUR_NVIDIA_API_KEY_HERE'")
        print("   OR set env var: $env:NVIDIA_API_KEY='your_key'")
        return

    if API_KEY.startswith("AIza"):
        print("\nERROR: You are using a Google Gemini Key (starts with AIza...)!")
        print("       Nvidia keys usually start with 'nvapi-'.")
        print("       Please use 'mode=ai' for Gemini, or get an Nvidia key.")
        return

    print(f"Analyzing {URL} using DeepSeek V3 (via Nvidia)...")
    print("This may take 15-30 seconds...")
    
    try:
        # Test the GET /clips endpoint with mode=nvidia
        resp = requests.get(f"{BASE_URL}/clips", params={
            "url": URL,
            "mode": "nvidia",
            "nvidia_key": API_KEY,
            "num": 3,
            "seconds": 30
        }, timeout=120)
        
        if resp.status_code == 200:
            data = resp.json()
            print("\nSUCCESS! AI Candidates Found (DeepSeek):")
            clips = data.get('clips', [])
            
            if not clips:
                print("   (No clips found. AI might have filtered everything or returned empty)")
                
            for i, clip in enumerate(clips):
                print(f"\nClip {i+1}: {clip.get('title')}")
                print(f"   Score: {clip.get('viral_score')}")
                print(f"   Download: {clip.get('download_link')}")
                print(f"   Reason: {clip.get('reason')}")
        else:
            print(f"\nError: {resp.status_code}")
            try:
                print(json.dumps(resp.json(), indent=2))
            except:
                print(resp.text)
            
    except Exception as e:
        print(f"\nException: {e}")

if __name__ == "__main__":
    test_nvidia()
