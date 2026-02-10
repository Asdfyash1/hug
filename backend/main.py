"""
Viral Clip Extractor API - FastAPI version for Hugging Face
"""
import os
import re
import json
import logging
import shutil
import tempfile
import hashlib
import uuid
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import requests

# Try to import AI clients
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Heuristic Mode Constants
DEFAULT_CLIP_LENGTH = 40
DEFAULT_WINDOW_STEP = 30
DEFAULT_MIN_CLIP_LENGTH = 25
DEFAULT_MAX_CLIP_LENGTH = 60
MAX_CANDIDATES = 10

HOOK_WORDS = [
    "wait", "shocking", "unbelievable", "secret", "crazy", "exposed", "omg", "wtf",
    "plot twist", "insane", "cheating", "stunning", "mind blowing", "truth", "lie",
    "revealed", "hidden", "discover", "amazing", "incredible", "must see", "urgent"
]

EMOTION_WORDS = [
    "laugh", "cry", "angry", "shocked", "surprised", "scream", "love", "hate",
    "disgusted", "horrified", "amazed", "excited", "sad", "happy", "furious"
]

# FastAPI app
app = FastAPI(
    title="Viral Clip Extractor API",
    description="Extract viral clips from YouTube videos using AI",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "/app/downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Ensure ffmpeg is available
FFMPEG_PATH = shutil.which("ffmpeg")
if FFMPEG_PATH:
    logger.info(f"FFmpeg found at: {FFMPEG_PATH}")
else:
    logger.warning("FFmpeg not found in PATH!")


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def parse_vtt_content(content: str) -> List[Dict]:
    """Parse VTT subtitle content into segments"""
    segments = []
    lines = content.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            time_match = re.match(r'(\d+:\d+:\d+\.\d+|\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+|\d+:\d+\.\d+)', line)
            if time_match:
                start_str, end_str = time_match.groups()
                
                def parse_time(t):
                    parts = t.replace(',', '.').split(':')
                    if len(parts) == 3:
                        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                    else:
                        return float(parts[0]) * 60 + float(parts[1])
                
                start = parse_time(start_str)
                end = parse_time(end_str)
                
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text_lines.append(re.sub(r'<[^>]+>', '', lines[i].strip()))
                    i += 1
                
                text = ' '.join(text_lines)
                if text:
                    segments.append({'start': start, 'end': end, 'text': text})
                continue
        i += 1
    return segments


class ViralClipExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.base_ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 3,
            'nocheckcertificate': True,
            'http_headers': self.headers,
        }
    
    def extract_video_info(self, url: str) -> Dict:
        """Extract basic video information"""
        logger.info(f"Extracting video info for: {url}")
        try:
            with yt_dlp.YoutubeDL(self.base_ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'thumbnail': info.get('thumbnail'),
                    'channel': info.get('channel'),
                    'view_count': info.get('view_count'),
                }
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            return {}
    
    def fetch_full_transcript(self, url: str) -> List[Dict]:
        """Fetch transcript from YouTube video"""
        logger.info(f"Fetching transcript for: {url}")
        ydl_opts = {
            **self.base_ydl_opts,
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                sub_url = None
                if 'en' in info.get('subtitles', {}):
                    sub_url = info['subtitles']['en'][-1].get('url')
                elif 'en' in info.get('automatic_captions', {}):
                    sub_url = info['automatic_captions']['en'][-1].get('url')
                
                if not sub_url:
                    logger.warning("No English subtitles found")
                    return []
                
                logger.info(f"Found subtitle URL: {sub_url[:80]}...")
                response = requests.get(sub_url, headers=self.headers, timeout=15)
                response.raise_for_status()
                sub_text = response.text
                logger.info(f"Downloaded {len(sub_text)} chars of subtitle")
                
                segments = []
                if 'vtt' in sub_url or sub_text.startswith('WEBVTT'):
                    segments = parse_vtt_content(sub_text)
                else:
                    # Try JSON format
                    try:
                        data = json.loads(sub_text)
                        if 'events' in data:
                            for event in data['events']:
                                if 'segs' in event:
                                    start_time = event.get('tStartMs', 0) / 1000
                                    duration = event.get('dDurationMs', 0) / 1000
                                    text = ''.join(seg.get('utf8', '') for seg in event['segs'])
                                    segments.append({
                                        'start': start_time,
                                        'end': start_time + duration,
                                        'text': text
                                    })
                    except json.JSONDecodeError:
                        segments = parse_vtt_content(sub_text)
                
                logger.info(f"Parsed {len(segments)} transcript segments")
                return segments
                
        except Exception as e:
            logger.error(f"Transcript error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_transcript_text(self, segments: List[Dict], start: float, end: float) -> str:
        """Filter transcript from pre-fetched segments"""
        relevant = [
            seg["text"] for seg in segments
            if seg["end"] >= start and seg["start"] <= end and seg["text"].strip()
        ]
        return " ".join(relevant)
    
    def score_clip(self, transcript_segments: List[Dict], start: float, end: float) -> Dict:
        """Calculate viral score for clip using heuristic analysis"""
        transcript = self.get_transcript_text(transcript_segments, start, end).lower()
        
        score = 0
        reasons = []
        
        # Hook words (up to 50 points)
        hooks = [w for w in HOOK_WORDS if w in transcript]
        hook_score = min(len(hooks) * 10, 50)
        score += hook_score
        if hooks:
            reasons.append(f"Hooks: {', '.join(hooks[:3])}")
        
        # Emotion words (up to 30 points)
        emotions = [w for w in EMOTION_WORDS if w in transcript]
        emotion_score = min(len(emotions) * 6, 30)
        score += emotion_score
        if emotions:
            reasons.append(f"Emotion: {', '.join(emotions[:2])}")
        
        # Conflict indicators (20 points)
        conflict_words = ["but", "however", "argument", "fight", "wrong", "disagree"]
        if any(w in transcript for w in conflict_words):
            score += 20
            reasons.append("Conflict detected")
        
        # Duration bonus
        duration = end - start
        if DEFAULT_MIN_CLIP_LENGTH <= duration <= DEFAULT_MAX_CLIP_LENGTH:
            score += 10
            reasons.append("Optimal duration")
        elif duration < DEFAULT_MIN_CLIP_LENGTH:
            score -= 10
            reasons.append("Short clip")
        elif duration > DEFAULT_MAX_CLIP_LENGTH:
            score -= 5
            reasons.append("Long clip")
        
        # Transcript quality bonus
        word_count = len(transcript.split())
        if word_count > 10:
            score += 5
            reasons.append("Good content density")
        
        return {
            "start": start,
            "end": end,
            "duration": round(duration, 1),
            "viral_score": max(0, min(score, 100)),
            "reason": "; ".join(reasons) if reasons else "Heuristic analysis",
        }
    
    def generate_candidates(self, video_duration: float, clip_length: int = None) -> List[Dict]:
        """Generate candidate clips using sliding window"""
        clip_length = clip_length or DEFAULT_CLIP_LENGTH
        candidates = []
        
        # Sliding windows
        for start in range(0, int(video_duration) - clip_length, DEFAULT_WINDOW_STEP):
            candidates.append({
                "start": start,
                "end": min(start + clip_length, video_duration)
            })
        
        return candidates[:MAX_CANDIDATES]  # Limit candidates for speed
    
    def analyze_with_nvidia(self, transcript_segments: List[Dict], api_key: str) -> List[Dict]:
        """Analyze transcript using Nvidia DeepSeek API"""
        if not HAS_OPENAI:
            raise ValueError("OpenAI package not installed")
        
        if not transcript_segments:
            return []
        
        # Combine segments into chunks
        full_text = ""
        for seg in transcript_segments:
            full_text += f"[{seg['start']:.1f}s] {seg['text']} "
        
        # Split into ~5 minute chunks
        chunk_size = 300  # seconds
        chunks = []
        current_chunk = []
        current_start = 0
        
        for seg in transcript_segments:
            if seg['start'] - current_start > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_start = seg['start']
            current_chunk.append(seg)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        all_clips = []
        
        def process_chunk(chunk_data):
            chunk_index, chunk = chunk_data
            chunk_text = " ".join([f"[{s['start']:.1f}s] {s['text']}" for s in chunk])
            
            prompt = f"""Analyze this transcript section and identify 1-2 viral-worthy clips.
Each clip should be 15-60 seconds and have high viral potential.

Return JSON array:
[{{"start": <seconds>, "end": <seconds>, "viral_score": <1-100>, "reason": "<why viral>"}}]

Transcript:
{chunk_text}

JSON response only:"""
            
            try:
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key
                )
                
                completion = client.chat.completions.create(
                    model="deepseek-ai/deepseek-r1-distill-llama-70b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                )
                
                text = completion.choices[0].message.content
                text = re.sub(r"```json\s*", "", text)
                text = re.sub(r"```\s*", "", text)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                
                return json.loads(text)
            except Exception as e:
                logger.error(f"Chunk {chunk_index} error: {e}")
                return []
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_chunk, (i, chunk)): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_clips.extend(result)
        
        # Sort by viral score
        all_clips.sort(key=lambda x: x.get('viral_score', 0), reverse=True)
        return all_clips[:10]  # Return top 10
    
    def analyze_with_gemini(self, transcript_segments: List[Dict], api_key: str) -> List[Dict]:
        """Analyze transcript using Gemini API"""
        if not HAS_GEMINI:
            raise ValueError("Google Generative AI package not installed")
        
        if not transcript_segments:
            return []
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        chunk_text = " ".join([f"[{s['start']:.1f}s] {s['text']}" for s in transcript_segments[:500]])
        
        prompt = f"""Analyze this transcript and identify 3-5 viral-worthy clips.
Each clip should be 15-60 seconds with high engagement potential.

Return JSON array only:
[{{"start": <seconds>, "end": <seconds>, "viral_score": <1-100>, "reason": "<why viral>"}}]

Transcript:
{chunk_text}

JSON:"""
        
        try:
            response = model.generate_content(prompt)
            text = response.text
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            clips = json.loads(text)
            clips.sort(key=lambda x: x.get('viral_score', 0), reverse=True)
            return clips[:10]
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return []
    
    def download_clip(self, url: str, start: float, end: float, quality: str = "480") -> Optional[str]:
        """Download and cut a clip"""
        video_id = extract_video_id(url)
        if not video_id:
            return None
        
        output_file = os.path.join(DOWNLOAD_DIR, f"{video_id}_{int(start)}_{int(end)}.mp4")
        
        if os.path.exists(output_file):
            return output_file
        
        format_str = f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best"
        output_template = os.path.join(DOWNLOAD_DIR, f"{video_id}_%(id)s.%(ext)s")
        
        ydl_opts = {
            **self.base_ydl_opts,
            'format': format_str,
            'outtmpl': output_template,
            'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
            'force_keyframes_at_cuts': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find downloaded file and rename
            for f in os.listdir(DOWNLOAD_DIR):
                if f.startswith(video_id) and f != os.path.basename(output_file):
                    src = os.path.join(DOWNLOAD_DIR, f)
                    os.rename(src, output_file)
                    break
            
            if os.path.exists(output_file):
                return output_file
            return None
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None


# Global extractor instance
extractor = ViralClipExtractor()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Viral Clip Extractor API", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "ffmpeg": FFMPEG_PATH is not None}


@app.get("/clips")
async def get_clips(
    url: str = Query(..., description="YouTube video URL"),
    mode: str = Query("nvidia", description="Analysis mode: nvidia, gemini, or heuristic"),
    nvidia_key: Optional[str] = Query(None, description="Nvidia API key"),
    gemini_key: Optional[str] = Query(None, description="Gemini API key"),
    api_key: Optional[str] = Query(None, description="Generic API key"),
    num: int = Query(5, description="Number of clips to return"),
):
    """Extract viral clips from a YouTube video"""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Get API key
        key = nvidia_key or gemini_key or api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        # Get video info
        video_info = extractor.extract_video_info(url)
        
        # Fetch transcript
        transcript = extractor.fetch_full_transcript(url)
        
        clips = []
        if transcript:
            if mode == "nvidia" and key:
                clips = extractor.analyze_with_nvidia(transcript, key)
            elif mode == "gemini" and key:
                clips = extractor.analyze_with_gemini(transcript, key)
            elif mode == "heuristic" or not key:
                # Heuristic mode (default when no API key)
                logger.info("Using heuristic mode for clip analysis")
                video_duration = video_info.get("duration", 0)
                if video_duration > 0:
                    candidates = extractor.generate_candidates(video_duration)
                    for cand in candidates:
                        scored = extractor.score_clip(transcript, cand['start'], cand['end'])
                        clips.append(scored)
                    # Sort by viral score
                    clips.sort(key=lambda x: x.get('viral_score', 0), reverse=True)
        
        return {
            "success": True,
            "video_id": video_id,
            "video_title": video_info.get("title"),
            "video_duration": video_info.get("duration"),
            "mode": mode,
            "clips": clips[:num],
            "clips_count": len(clips[:num]),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clips error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/debug_video")
async def debug_video(
    url: str = Query(..., description="YouTube video URL"),
):
    """Debug endpoint to check transcript fetching"""
    try:
        video_id = extract_video_id(url)
        video_info = extractor.extract_video_info(url)
        transcript = extractor.fetch_full_transcript(url)
        
        return {
            "video_id": video_id,
            "video_title": video_info.get("title"),
            "video_duration": video_info.get("duration"),
            "transcript_segments_count": len(transcript),
            "transcript_sample": transcript[:5] if transcript else [],
        }
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )


@app.get("/download")
async def download_clip(
    url: str = Query(..., description="YouTube video URL"),
    start: float = Query(..., description="Start time in seconds"),
    end: float = Query(..., description="End time in seconds"),
    quality: str = Query("480", description="Video quality"),
):
    """Download a specific clip"""
    try:
        clip_path = extractor.download_clip(url, start, end, quality)
        if clip_path and os.path.exists(clip_path):
            return FileResponse(
                clip_path,
                media_type="video/mp4",
                filename=os.path.basename(clip_path)
            )
        raise HTTPException(status_code=404, detail="Failed to download clip")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
