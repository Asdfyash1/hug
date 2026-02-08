
import requests
import time
import os

BASE_URL = "http://127.0.0.1:5000"

def test_e2e_download():
    print("Testing /clips and download link...")
    url = "https://youtu.be/uVkFrqugXFQ" 
    
    try:
        # 1. Get Clips
        print("Fetching clips...")
        resp = requests.get(f"{BASE_URL}/clips", params={
            "url": url,
            "seconds": 5, # Short clip for speed
            "num": 1
        }, timeout=60)
        
        if resp.status_code != 200:
            print(f"Failed to get clips: {resp.text}")
            return
            
        data = resp.json()
        clips = data.get('clips', [])
        if not clips:
            print("No clips found in response")
            return
            
        clip = clips[0]
        download_link = clip.get('download_link')
        if not download_link:
            print("No download_link in clip data")
            print(clip)
            return
            
        print(f"Found download link: {download_link}")
        
        # 2. Download File
        full_download_url = f"{BASE_URL}{download_link}"
        print(f"Downloading from {full_download_url}...")
        
        t0 = time.time()
        file_resp = requests.get(full_download_url, stream=True, timeout=120)
        
        if file_resp.status_code == 200:
            filename = "test_clip_downloaded.mp4"
            with open(filename, 'wb') as f:
                for chunk in file_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size = os.path.getsize(filename)
            print(f"Downloaded file: {filename} ({size/1024:.2f} KB)")
            print(f"Time: {time.time() - t0:.2f}s")
        else:
            print(f"Failed to download file: {file_resp.status_code}")
            print(file_resp.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_e2e_download()
