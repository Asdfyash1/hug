"""
Viral Clip Extractor - Python Client Example
Usage: python python-client.py
"""

import requests
import json

# Configuration
API_BASE_URL = "https://your-api.vercel.app"  # Replace with your API URL


def analyze_video(url: str, num_clips: int = 5, clip_length: int = 40, quality: str = "720"):
    """
    Analyze a YouTube video for viral clips
    
    Args:
        url: YouTube video URL
        num_clips: Number of clips to return
        clip_length: Target clip length in seconds
        quality: Video quality (360, 720, 1080)
    """
    endpoint = f"{API_BASE_URL}/analyze"
    
    payload = {
        "url": url,
        "num_clips": num_clips,
        "clip_length": clip_length,
        "quality": quality
    }
    
    print(f"🔍 Analyzing: {url}")
    print(f"   Clips: {num_clips} | Length: {clip_length}s | Quality: {quality}p")
    print("-" * 60)
    
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            print(f"❌ Error: {data.get('error')}")
            return None
        
        print(f"✅ Found {len(data['clips'])} viral clips!")
        print(f"📹 Video: {data['video_title']}")
        print()
        
        for i, clip in enumerate(data['clips'], 1):
            print(f"CLIP #{i}")
            print(f"  Score: {clip['viral_score']}% viral")
            print(f"  Time: {clip['start_formatted']} - {clip['end_formatted']}")
            print(f"  Reasons: {', '.join(clip['reasons'])}")
            print(f"  Preview: {clip['transcript_preview'][:80]}...")
            print(f"  URL: {clip['youtube_url']}")
            print()
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def extract_video_info(url: str):
    """Extract basic video information"""
    endpoint = f"{API_BASE_URL}/extract"
    
    print(f"📹 Extracting info: {url}")
    
    try:
        response = requests.post(endpoint, json={"url": url}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            info = data["data"]
            print(f"✅ Title: {info['title']}")
            print(f"   Channel: {info['uploader']}")
            print(f"   Duration: {info['duration']}s")
            print(f"   Views: {info.get('view_count', 'N/A')}")
            return info
        else:
            print(f"❌ Error: {data.get('error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def get_transcript(url: str, start: int = 0, end: int = 60):
    """Get transcript for a specific time range"""
    endpoint = f"{API_BASE_URL}/transcript"
    
    payload = {
        "url": url,
        "start": start,
        "end": end
    }
    
    print(f"📝 Getting transcript: {start}s - {end}s")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            print(f"✅ Transcript ({data['word_count']} words):")
            print(f"   {data['transcript'][:200]}...")
            return data
        else:
            print(f"❌ Error: {data.get('error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def score_clip(url: str, start: str, end: str):
    """Score a specific time range for viral potential"""
    endpoint = f"{API_BASE_URL}/score"
    
    payload = {
        "url": url,
        "start": start,
        "end": end
    }
    
    print(f"🎯 Scoring clip: {start} - {end}")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            clip = data["clip"]
            print(f"✅ Viral Score: {clip['viral_score']}%")
            print(f"   Reasons: {', '.join(clip['reasons'])}")
            print(f"   Words: {clip['word_count']}")
            return clip
        else:
            print(f"❌ Error: {data.get('error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def get_download_url(url: str, start: str, end: str, quality: str = "720"):
    """Get download URL for a clip"""
    endpoint = f"{API_BASE_URL}/download"
    
    payload = {
        "url": url,
        "start": start,
        "end": end,
        "quality": quality
    }
    
    print(f"⬇️ Getting download URL: {start} - {end}")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            info = data["download_info"]
            print(f"✅ Download ready!")
            print(f"   URL: {info['download_url'][:80]}...")
            print(f"   Formats available: {len(info['formats'])}")
            return info
        else:
            print(f"❌ Error: {data.get('error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def batch_analyze(urls: list, num_clips: int = 3):
    """Analyze multiple videos in batch"""
    results = []
    
    print(f"🔄 Batch analyzing {len(urls)} videos...")
    print("=" * 60)
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]")
        result = analyze_video(url, num_clips=num_clips)
        if result:
            results.append(result)
    
    print("\n" + "=" * 60)
    print(f"✅ Batch complete! Analyzed {len(results)} videos")
    
    return results


# Example usage
if __name__ == "__main__":
    # Replace with your API URL
    API_BASE_URL = input("Enter your API base URL: ").strip()
    
    # Example YouTube URL (replace with actual URL)
    test_url = input("Enter YouTube URL: ").strip()
    
    print("\n" + "=" * 60)
    print("VIRAL CLIP EXTRACTOR - Python Client")
    print("=" * 60 + "\n")
    
    # Menu
    print("Choose an option:")
    print("1. Full Analysis (find viral clips)")
    print("2. Extract Video Info")
    print("3. Get Transcript")
    print("4. Score Specific Range")
    print("5. Get Download URL")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        num = int(input("Number of clips (default 5): ") or "5")
        length = int(input("Clip length in seconds (default 40): ") or "40")
        analyze_video(test_url, num_clips=num, clip_length=length)
    
    elif choice == "2":
        extract_video_info(test_url)
    
    elif choice == "3":
        start = int(input("Start time in seconds: "))
        end = int(input("End time in seconds: "))
        get_transcript(test_url, start, end)
    
    elif choice == "4":
        start = input("Start time (MM:SS or seconds): ")
        end = input("End time (MM:SS or seconds): ")
        score_clip(test_url, start, end)
    
    elif choice == "5":
        start = input("Start time (MM:SS or seconds): ")
        end = input("End time (MM:SS or seconds): ")
        quality = input("Quality (360/720/1080, default 720): ") or "720"
        get_download_url(test_url, start, end, quality)
    
    else:
        print("Invalid choice!")
