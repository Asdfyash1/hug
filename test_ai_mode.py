
import requests
import os
import json
import sys

# REPLACE THIS WITH YOUR KEY or set GEMINI_API_KEY env var
# Get from: https://aistudio.google.com/app/apikey
DEFAULT_KEY = "AIzaSyAOXzvcFroHyNa5UtYRi3c8AGX0MJ1Tb5E"
API_KEY = os.environ.get("GEMINI_API_KEY", DEFAULT_KEY) 

URL = "https://youtu.be/uVkFrqugXFQ"
BASE_URL = "http://127.0.0.1:5000"

def test_ai():
    print("Testing AI Mode...")
    
    # Check if key is set
    if "YOUR_GEMINI" in API_KEY:
        print("\nWARNING: You haven't set your Gemini API Key!")
        print("1. Get a key here: https://aistudio.google.com/app/apikey")
        print("2. Open this script and replace 'YOUR_GEMINI_API_KEY_HERE'")
        print("   OR set env var: $env:GEMINI_API_KEY='your_key'")
        return

    print(f"Analyzing {URL} using Gemini...")
    print("This may take 10-20 seconds...")
    
    try:
        # Test the GET /clips endpoint with mode=ai
        resp = requests.get(f"{BASE_URL}/clips", params={
            "url": URL,
            "mode": "ai",
            "gemini_key": API_KEY,
            "num": 3,
            "seconds": 30
        }, timeout=120)
        
        if resp.status_code == 200:
            data = resp.json()
            print("\nSUCCESS! AI Candidates Found:")
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
    test_ai()
