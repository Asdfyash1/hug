import logging
import json
from api.index import ViralClipExtractor, extract_video_id

logging.basicConfig(level=logging.INFO)

def test():
    url = "https://youtu.be/uVkFrqugXFQ"
    nvidia_key = "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"
    
    print("=== LOCAL TEST WITH FIXED MODEL ===")
    print(f"URL: {url}")
    
    extractor = ViralClipExtractor()
    
    # 1. Video Info
    print("\n1. Fetching Video Info...")
    video_info = extractor.extract_video_info(url)
    print(f"   Title: {video_info.get('title')}")
    print(f"   Duration: {video_info.get('duration')} seconds")
    
    # 2. Transcript
    print("\n2. Fetching Transcript...")
    transcript = extractor.fetch_full_transcript(url)
    print(f"   Transcript Segments: {len(transcript)}")
    if not transcript:
        print("   [FAIL] No transcript found!")
        return
    
    # 3. AI Analysis
    print("\n3. Running Nvidia AI Analysis with deepseek-v3.2...")
    clips = extractor.analyze_with_nvidia(transcript, nvidia_key)
    print(f"   Clips Found: {len(clips)}")
    
    # 4. Final Response
    response = {
        "success": True,
        "video_id": extract_video_id(url),
        "clips": clips,
        "clips_count": len(clips)
    }
    
    print("\n=== FINAL RESPONSE ===")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    test()
