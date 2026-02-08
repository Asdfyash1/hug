
import urllib.request
import zipfile
import os
import shutil

FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
DOWNLOAD_path = "ffmpeg.zip"
EXTRACT_DIR = "ffmpeg_temp"
FINAL_DIR = "ffmpeg"

def install_ffmpeg():
    print(f"Downloading FFmpeg from {FFMPEG_URL}...")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, DOWNLOAD_path)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(DOWNLOAD_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        
        # Move bin folder to top level
        # The zip structure is usually ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe
        # We want ffmpeg/bin/ffmpeg.exe
        
        extracted_root = os.path.join(EXTRACT_DIR, os.listdir(EXTRACT_DIR)[0])
        
        if os.path.exists(FINAL_DIR):
            print("Removing old installation...")
            shutil.rmtree(FINAL_DIR)
            
        shutil.move(extracted_root, FINAL_DIR)
        
        # Cleanup
        os.remove(DOWNLOAD_path)
        shutil.rmtree(EXTRACT_DIR)
        
        print(f"FFmpeg installed to {os.path.abspath(FINAL_DIR)}")
        
        # Verify
        bin_path = os.path.join(FINAL_DIR, "bin", "ffmpeg.exe")
        if os.path.exists(bin_path):
            print(f"✅ Verified: {bin_path}")
        else:
            print("❌ Error: ffmpeg.exe not found in expected path")
            
    except Exception as e:
        print(f"Error installing FFmpeg: {e}")

if __name__ == "__main__":
    install_ffmpeg()
