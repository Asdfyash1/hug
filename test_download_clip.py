
from api.index import ViralClipExtractor
import os

def test_download():
    extractor = ViralClipExtractor()
    url = "https://youtu.be/uVkFrqugXFQ"
    print("Testing download of 5s clip...")
    try:
        # Try to download 5s segment
        path = extractor.download_clip(url, "0:00", "0:05") # Method doesn't exist yet in my class, I need to add it or test raw yt-dlp
        print(f"Downloaded to: {path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # monkey patch for testing purposes or just write raw yt-dlp code here
    import yt_dlp
    
    url = "https://youtu.be/uVkFrqugXFQ"
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best',
        'outtmpl': 'test_clip.%(ext)s',
        'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 5}],
        'force_keyframes_at_cuts': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download success")
    except Exception as e:
        print(f"Download failed: {e}")
