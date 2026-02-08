"""
Test YouTube bot detection fix
"""
import sys
import os
sys.path.append(os.getcwd())

from api.index import ViralClipExtractor

def test_youtube_access():
    print("Testing YouTube access with updated configuration...")
    print("=" * 60)
    
    url = "https://youtu.be/uVkFrqugXFQ"
    
    try:
        extractor = ViralClipExtractor()
        print(f"\n1. Extracting video info for: {url}")
        video_info = extractor.extract_video_info(url)
        
        print(f"[OK] Video Title: {video_info.get('title')}")
        print(f"[OK] Duration: {video_info.get('duration')} seconds")
        print(f"[OK] Uploader: {video_info.get('uploader')}")
        
        print(f"\n2. Fetching transcript...")
        transcript = extractor.fetch_full_transcript(url)
        
        if transcript:
            print(f"[OK] Transcript segments: {len(transcript)}")
            print(f"[OK] First segment: {transcript[0] if transcript else 'N/A'}")
            print("\n[SUCCESS] YouTube bot detection bypassed!")
        else:
            print("[WARN] No transcript found (might be disabled for this video)")
            print("[SUCCESS] Video info extracted successfully!")
            
    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_youtube_access()
    sys.exit(0 if success else 1)
