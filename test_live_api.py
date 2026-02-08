import requests
import sys
import json

url = "https://kimi-agent-viral-clip-extractor.onrender.com/clips"
params = {
    "url": "https://youtu.be/uVkFrqugXFQ",
    "mode": "nvidia",
    "nvidia_key": "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",
    "num": 3
}

print(f"Testing live API endpoint...")
print(f"URL: {url}")
print(f"Params: {json.dumps({k: v if k != 'nvidia_key' else '***' for k, v in params.items()}, indent=2)}")
print("=" * 60)

try:
    response = requests.get(url, params=params, timeout=120)
    print(f"\nStatus Code: {response.status_code}")
    
    try:
        data = response.json()
        print("\nResponse JSON:")
        print(json.dumps(data, indent=2))
        
        if data.get("success"):
            print(f"\n[SUCCESS] API working! Found {len(data.get('clips', []))} clips")
            sys.exit(0)
        else:
            print(f"\n[FAILED] API returned error: {data.get('error')}")
            sys.exit(1)
    except:
        print("\nResponse Text:")
        print(response.text)
        sys.exit(1)
        
except Exception as e:
    print(f"\n[ERROR] {e}")
    sys.exit(1)
