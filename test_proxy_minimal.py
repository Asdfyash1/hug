"""
Minimal proxy test - only fetches metadata, no downloads
Bandwidth usage: ~10-50KB per test
"""
import sys
import os
sys.path.append(os.getcwd())

# Test with first proxy from the list
PROXY = "31.59.20.176:6754:nntlrciu:sx2noxvkj6y7"
host_port, user, password = PROXY.rsplit(':', 2)
proxy_url = f"http://{user}:{password}@{host_port}"

# Set environment variable
os.environ['PROXY_URL'] = proxy_url

from api.index import ViralClipExtractor

def test_proxy_minimal():
    print("=" * 60)
    print("MINIMAL PROXY TEST (Metadata Only - Low Bandwidth)")
    print("=" * 60)
    print(f"Proxy: {host_port}")
    print(f"Test URL: https://youtu.be/uVkFrqugXFQ")
    print("=" * 60)
    
    url = "https://youtu.be/uVkFrqugXFQ"
    
    try:
        extractor = ViralClipExtractor()
        
        # Only fetch video info (minimal bandwidth)
        print("\n[1/2] Fetching video metadata...")
        video_info = extractor.extract_video_info(url)
        
        print(f"[OK] Video Title: {video_info.get('title')[:50]}...")
        print(f"[OK] Duration: {video_info.get('duration')} seconds")
        print(f"[OK] Uploader: {video_info.get('uploader')}")
        
        # Fetch transcript (also minimal bandwidth)
        print("\n[2/2] Fetching transcript...")
        transcript = extractor.fetch_full_transcript(url)
        
        if transcript and len(transcript) > 0:
            print(f"[OK] Transcript segments: {len(transcript)}")
            print(f"[OK] First segment: {transcript[0]}")
            print("\n" + "=" * 60)
            print("[SUCCESS] Proxy works! YouTube bot detection bypassed!")
            print("=" * 60)
            return True
        else:
            print("[WARN] No transcript, but video info fetched successfully")
            print("\n" + "=" * 60)
            print("[SUCCESS] Proxy works!")
            print("=" * 60)
            return True
            
    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_proxy_minimal()
    sys.exit(0 if success else 1)
