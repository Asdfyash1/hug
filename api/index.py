"""
Viral Clip Extractor API
Deploy to Vercel for YouTube viral clip analysis
"""

from flask import Flask, request, jsonify, send_file, after_this_request
from flask_cors import CORS
import yt_dlp
import requests
import google.generativeai as genai
from openai import OpenAI
import json
import re
from io import StringIO
import os
import re
from typing import Dict, List, Optional, Any

app = Flask(__name__)
CORS(app)

# ============== CONFIGURATION ==============
DEFAULT_CLIP_LENGTH = 40
DEFAULT_WINDOW_STEP = 10
DEFAULT_MIN_CLIP_LENGTH = 25
DEFAULT_MAX_CLIP_LENGTH = 60

HOOK_WORDS = [
    "wait", "shocking", "unbelievable", "secret", "crazy", "exposed", "omg", "wtf",
    "plot twist", "insane", "cheating", "stunning", "mind blowing", "truth", "lie",
    "revealed", "hidden", "discover", "amazing", "incredible", "must see", "urgent"
]

EMOTION_WORDS = [
    "laugh", "cry", "angry", "shocked", "surprised", "scream", "love", "hate",
    "disgusted", "horrified", "amazed", "excited", "sad", "happy", "furious"
]

# ============== UTILITY FUNCTIONS ==============

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def time_to_seconds(t) -> float:
    """Convert time string to seconds"""
    if isinstance(t, (int, float)):
        return float(t)
    if ':' in str(t):
        parts = list(map(int, str(t).split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 1:
            return parts[0]
    return float(t)

def seconds_to_time(s: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(s // 60)
    seconds = int(s % 60)
    return f"{minutes}:{seconds:02d}"

def parse_vtt_content(vtt_text: str) -> List[Dict]:
    """Parse VTT subtitle content"""
    segments = []
    lines = vtt_text.strip().split('\n')
    
    i = 0
    while i < len(lines) and not '-->' in lines[i]:
        i += 1
    
    current_text = []
    current_start = None
    current_end = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        if '-->' in line:
            if current_text and current_start is not None:
                segments.append({
                    'start': current_start,
                    'end': current_end,
                    'text': ' '.join(current_text)
                })
            
            times = line.split('-->')
            if len(times) == 2:
                start_str = times[0].strip().split('.')[0]
                end_str = times[1].strip().split()[0].split('.')[0]
                
                current_start = time_to_seconds(start_str)
                current_end = time_to_seconds(end_str)
                current_text = []
        elif line and not line.startswith('NOTE') and not line.startswith('STYLE'):
            clean_line = re.sub(r'<[^>]+>', '', line)
            if clean_line:
                current_text.append(clean_line)
        
        i += 1
    
    if current_text and current_start is not None:
        segments.append({
            'start': current_start,
            'end': current_end,
            'text': ' '.join(current_text)
        })
    
    return segments

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== CORE CLASS ==============

class ViralClipExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        # Robust FFmpeg path resolution
        import shutil
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Check local ffmpeg folder (Windows/Dev)
        ffmpeg_path_local = os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe")
        
        # 2. Check system PATH (Linux/Render/Docker)
        ffmpeg_path_system = shutil.which("ffmpeg")
        
        ffmpeg_location = None
        if os.path.exists(ffmpeg_path_local):
            ffmpeg_location = ffmpeg_path_local
            ffmpeg_dir = os.path.dirname(ffmpeg_location)
            if ffmpeg_dir not in os.environ["PATH"]:
                 os.environ["PATH"] += os.pathsep + ffmpeg_dir
            logger.info(f"Using Local FFmpeg: {ffmpeg_location}")
        elif ffmpeg_path_system:
            # If on system path, we don't need to specify location for yt-dlp usually, 
            # but getting the dir is good practice
            ffmpeg_location = ffmpeg_path_system
            logger.info(f"Using System FFmpeg: {ffmpeg_location}")
        else:
            logger.warning("FFmpeg NOT FOUND! Clip cutting will fail.")


        self.base_ydl_opts = {
            'quiet': False,
            'no_warnings': False,
            # 'extractor_args': {'youtube': {'player_client': ['web']}},
            'http_headers': self.headers,
            # 'ffmpeg_location': ffmpeg_dir # Removed, relying on PATH
        }
        
        if ffmpeg_dir:
            logger.info(f"Added FFmpeg to PATH: {ffmpeg_dir}")
        else:
            logger.warning("FFmpeg NOT FOUND! Clip cutting will fail.")
    
    def download_clip(self, url: str, start: float, end: float, quality: str = "720") -> Optional[str]:
        """Download and cut a clip"""
        import uuid
        output_dir = "downloads"
        os.makedirs(output_dir, exist_ok=True)
        clip_id = f"clip_{uuid.uuid4().hex[:8]}"
        output_template = os.path.join(output_dir, f"{clip_id}.%(ext)s")
        
        ydl_opts = {
            **self.base_ydl_opts,
            'format': f'bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best',
            'outtmpl': output_template,
            'download_ranges': lambda info, ydl: [{
                'start_time': start,
                'end_time': end
            }],
            'force_keyframes_at_cuts': True, 
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the downloaded file
            for file in os.listdir(output_dir):
                if file.startswith(clip_id):
                    return os.path.abspath(os.path.join(output_dir, file))
            return None
        except Exception as e:
            with open("error.log", "a") as f:
                f.write(f"Download Error: {e}\n")
            logger.error(f"Error downloading clip: {e}")
            return None

    def analyze_with_gemini(self, transcript_segments: List[Dict], api_key: str) -> List[Dict]:
        """Analyze full transcript using Gemini AI"""
        try:
            genai.configure(api_key=api_key)
            # Using 2.0 Flash as 1.5 is unavailable for this key/region
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Helper to check if API key is likely valid (heuristic)
            if "YOUR_GEMINI" in api_key:
                raise ValueError("Invalid API Key placeholder used")

            # Construct prompt
            formatted_text = ""
            for seg in transcript_segments:
                start = int(seg['start'])
                text = seg['text']
                formatted_text += f"[{start}s] {text} "
            
            # Truncate
            full_context = formatted_text[:30000] # Safe limit

            prompt = f"""
            You are a viral content expert. Analyze the video transcript (with timestamps) and identify the top 3-5 most engaging clips for Shorts/TikTok.
            
            Transcript:
            {full_context}
            
            Return ONLY a raw JSON array (no markdown) of objects:
            [
                {{
                    "start": <number_seconds>,
                    "end": <number_seconds>,
                    "viral_score": <0-100>,
                    "reason": "<short explanation>"
                }}
            ]
            """
            
            logger.info("Sending request to Gemini AI...")
            response = model.generate_content(prompt)
            text = response.text
            # Clean markdown
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise e # Create visibility for the error

    def analyze_with_nvidia(self, transcript_segments: List[Dict], api_key: str) -> List[Dict]:
        """Analyze full transcript using Nvidia/DeepSeek AI"""
        try:
            client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key = api_key
            )

            # Construct prompt
            formatted_text = ""
            for seg in transcript_segments:
                start = int(seg['start'])
                text = seg['text']
                formatted_text += f"[{start}s] {text} "
            
            full_context = formatted_text[:30000]

            prompt = f"""
            You are a viral content expert. Analyze the video transcript (with timestamps) and identify the top 3-5 most engaging clips for Shorts/TikTok.
            
            Transcript:
            {full_context}
            
            Return ONLY a raw JSON array (no markdown) of objects:
            [
                {{
                    "start": <number_seconds>,
                    "end": <number_seconds>,
                    "viral_score": <0-100>,
                    "reason": "<short explanation>"
                }}
            ]
            """
            
            logger.info("Sending request to Nvidia/DeepSeek AI...")
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-v3.1-terminus",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"thinking":True}},
                stream=False
            )
            
            text = completion.choices[0].message.content
            # Clean markdown
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            # Remove thinking content
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            
            return json.loads(text)
        except Exception as e:
            logger.error(f"Nvidia API Error: {e}")
            raise e

    def extract_video_info(self, url: str) -> Dict:
        """Get video metadata"""
        logger.info(f"Extracting video info for: {url}")
        ydl_opts = {
            **self.base_ydl_opts,
            'simulate': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                chapters = []
                if 'chapters' in info and info['chapters']:
                    for ch in info['chapters']:
                        chapters.append({
                            "start": ch['start_time'],
                            "end": ch.get('end_time') or ch['start_time'] + DEFAULT_CLIP_LENGTH,
                            "title": ch.get('title', 'Untitled')
                        })
                
                logger.info(f"Video info extracted: {info.get('title')}")
                return {
                    "id": info.get("id"),
                    "title": info.get("title"),
                    "uploader": info.get("uploader"),
                    "duration": info.get("duration"),
                    "description": info.get("description", "")[:500],
                    "view_count": info.get("view_count"),
                    "like_count": info.get("like_count"),
                    "chapters": chapters
                }
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            raise
    
    def fetch_full_transcript(self, url: str) -> List[Dict]:
        """Fetch full transcript once"""
        logger.info(f"Fetching full transcript for: {url}")
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
                    logger.warning("No transcript found")
                    return []
                
                response = requests.get(sub_url, headers=self.headers, timeout=15)
                response.raise_for_status()
                sub_text = response.text
                
                segments = []
                if sub_url.endswith('.vtt') or 'vtt' in sub_url:
                    segments = parse_vtt_content(sub_text)
                elif sub_url.endswith('.srt') or 'srt' in sub_url:
                    # Reuse vtt logic as simple fallback
                     segments = parse_vtt_content(sub_text)
                
                # If we have segments, return them
                if segments:
                    return segments
                
                # Fallback to json3 if available/needed
                try:
                    import json
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
                except:
                    pass
                    
                return segments
                
        except Exception as e:
            logger.error(f"Transcript error: {e}")
            return []
    
    def get_transcript_text(self, segments: List[Dict], start: float, end: float) -> str:
        """Filter transcript from pre-fetched segments"""
        relevant = [
            seg["text"] for seg in segments
            if seg["end"] >= start and seg["start"] <= end and seg["text"].strip()
        ]
        return " ".join(relevant)
    
    def get_transcript(self, url: str, start: float, end: float) -> str:
        """Legacy method for backward compatibility"""
        segments = self.fetch_full_transcript(url)
        return self.get_transcript_text(segments, start, end)

    def score_clip(self, url: str, start: float, end: float, transcript_segments: List[Dict] = None) -> Dict:
        """Calculate viral score for clip"""
        if transcript_segments:
            transcript = self.get_transcript_text(transcript_segments, start, end).lower()
        else:
            # Fallback
            transcript = self.get_transcript(url, start, end).lower()
        
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
            "reasons": reasons,
            "transcript_preview": transcript[:150] + "..." if len(transcript) > 150 else transcript,
            "word_count": word_count
        }
    
    def generate_candidates(self, video_info: Dict, clip_length: int = None, window_step: int = None) -> List[Dict]:
        """Generate candidate clips"""
        duration = video_info.get("duration", 0)
        clip_length = clip_length or DEFAULT_CLIP_LENGTH
        window_step = window_step or DEFAULT_WINDOW_STEP
        candidates = []
        
        # Use chapters if available
        if video_info.get("chapters"):
            logger.info(f"Using {len(video_info['chapters'])} chapters for candidates")
            for ch in video_info["chapters"]:
                clip_start = ch["start"]
                clip_end = min(ch["end"], clip_start + clip_length)
                if clip_end - clip_start >= DEFAULT_MIN_CLIP_LENGTH:
                    candidates.append({
                        "start": clip_start,
                        "end": clip_end,
                        "title": ch.get("title", "Untitled")
                    })
        else:
            # Sliding windows
            logger.info("Using sliding windows for candidates")
            for start in range(0, int(duration) - clip_length, window_step):
                candidates.append({
                    "start": start,
                    "end": min(start + clip_length, duration)
                })
        
        return candidates

# ============== API ROUTES ==============

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API info"""
    return jsonify({
        "name": "Viral Clip Extractor API",
        "version": "1.3.0 (AI Enabled)",
        "description": "Extract and analyze viral clips from YouTube videos",
        "quick_start": {
            "endpoint": "/clips",
            "method": "GET",
            "usage": "/clips?url=YOUTUBE_URL&quality=720&seconds=40&num=5&mode=ai&gemini_key=YOUR_KEY",
            "example": "/clips?url=https://youtube.com/watch?v=dQw4w9WgXcQ&seconds=30&num=3"
        },
        "endpoints": {
            "/clips": "GET - Extract clips (Heuristic or AI)",
            "/health": "Health check",
            "/analyze": "POST - Full analysis",
            "/extract": "POST - Extract video info only",
            "/download_file": "GET - Download clip file"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "viral-clip-extractor"
    })

@app.route('/clips', methods=['GET'])
def get_clips():
    """
    Get viral clips (Heuristic, Gemini AI, or Nvidia AI)
    """
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({"error": "Missing 'url' parameter"}), 400
        
        quality = request.args.get('quality', '720')
        seconds = int(request.args.get('seconds', DEFAULT_CLIP_LENGTH))
        num_clips = int(request.args.get('num', 5))
        
        mode = request.args.get('mode', 'heuristic') # 'heuristic', 'ai' (defaults to gemini), 'nvidia'
        ai_provider = request.args.get('ai_provider', 'gemini') # 'gemini' or 'nvidia'
        
        # Handle API Keys
        nvidia_key = request.args.get('nvidia_key') or os.environ.get('NVIDIA_API_KEY')
        gemini_key = request.args.get('gemini_key') or os.environ.get('GEMINI_API_KEY')
        
        # Unified key handling if user passes 'api_key' generic param
        generic_key = request.args.get('api_key')
        if generic_key:
            if mode == 'nvidia' or ai_provider == 'nvidia':
                nvidia_key = generic_key
            else:
                gemini_key = generic_key

        extractor = ViralClipExtractor()
        
        # 1. Get video info
        video_info = extractor.extract_video_info(url)
        video_id = extract_video_id(url)
        
        if not video_id:
             return jsonify({"error": "Invalid YouTube URL"}), 400

        # 2. Fetch transcript ONCE
        transcript_segments = extractor.fetch_full_transcript(url)
        
        scored_clips = []
        
        if mode == 'ai' or mode == 'nvidia' or request.args.get('ai_provider'):
            # Determine provider
            provider = 'nvidia' if (mode == 'nvidia' or ai_provider == 'nvidia') else 'gemini'
            
            try:
                if provider == 'nvidia':
                    if not nvidia_key:
                         return jsonify({"error": "Nvidia API Key required"}), 400
                    logger.info("Using Nvidia DeepSeek for analysis...")
                    scored_clips = extractor.analyze_with_nvidia(transcript_segments, nvidia_key)
                else:
                    if not gemini_key:
                         return jsonify({"error": "Gemini API Key required"}), 400
                    logger.info("Using Gemini AI for analysis...")
                    scored_clips = extractor.analyze_with_gemini(transcript_segments, gemini_key)
                
                # Normalize response
                for clip in scored_clips:
                    clip['title'] = clip.get('reason', f'{provider.title()} Selected Clip')
                    clip.setdefault('start', 0)
                    clip.setdefault('end', clip['start'] + seconds)
                    clip.setdefault('viral_score', 80)
                    
            except Exception as e:
                return jsonify({"error": f"AI Analysis Failed ({provider}): {str(e)}"}), 500
        else:
            # Heuristic
            candidates = extractor.generate_candidates(video_info, seconds, DEFAULT_WINDOW_STEP)
            
            logger.info(f"Scoring {len(candidates)} candidates...")
            for i, cand in enumerate(candidates):
                score_data = extractor.score_clip(url, cand['start'], cand['end'], transcript_segments)
                score_data['title'] = cand.get('title', f'Clip {i+1}')
                scored_clips.append(score_data)
            
            scored_clips.sort(key=lambda x: x['viral_score'], reverse=True)
        
        # Select top non-overlapping clips
        final_clips = []
        used_ranges = []
        
        for clip in scored_clips:
            overlaps = False
            for used in used_ranges:
                if not (clip['end'] <= used['start'] or clip['start'] >= used['end']):
                    overlaps = True
                    break
            
            if not overlaps:
                final_clips.append(clip)
                used_ranges.append(clip)
                
                if len(final_clips) >= num_clips:
                    break
        
        import urllib.parse
        encoded_url = urllib.parse.quote(url)
        
        for clip in final_clips:
            # Ensure start/end are floats
            clip['start'] = float(clip['start'])
            clip['end'] = float(clip['end'])
            clip['start_formatted'] = seconds_to_time(clip['start'])
            clip['end_formatted'] = seconds_to_time(clip['end'])
            clip['youtube_url'] = f"https://youtube.com/watch?v={video_id}&t={int(clip['start'])}"
            clip['download_link'] = f"/download_file?url={encoded_url}&start={clip['start']}&end={clip['end']}&quality={quality}"
            
        return jsonify({
            "success": True,
            "video_id": video_id,
            "video_title": video_info.get('title'),
            "video_duration": video_info.get('duration'),
            "mode": mode,
            "clips": final_clips,
            "clips_count": len(final_clips)
        })
        
    except Exception as e:
        logger.error(f"Error in /clips: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/extract', methods=['POST'])
def extract_video():
    """Extract basic video information"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing 'url' parameter"}), 400
        
        url = data['url']
        video_id = extract_video_id(url)
        
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400
        
        extractor = ViralClipExtractor()
        info = extractor.extract_video_info(url)
        
        return jsonify({
            "success": True,
            "video_id": video_id,
            "data": info
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Full viral clip analysis"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing 'url' parameter"}), 400
        
        url = data['url']
        video_id = extract_video_id(url)
        
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400
        
        num_clips = data.get('num_clips', 5)
        clip_length = data.get('clip_length', DEFAULT_CLIP_LENGTH)
        window_step = data.get('window_step', DEFAULT_WINDOW_STEP)
        quality = data.get('quality', '720')
        mode = data.get('mode', 'heuristic')
        gemini_key = data.get('gemini_key') or os.environ.get('GEMINI_API_KEY')
        
        extractor = ViralClipExtractor()
        
        # 1. Get video info
        video_info = extractor.extract_video_info(url)
        
        # 2. Fetch transcript ONCE
        transcript_segments = extractor.fetch_full_transcript(url)
        
        if mode == 'ai':
            if not gemini_key:
                return jsonify({"error": "Gemini API Key required for AI mode"}), 400
            
            logger.info("Using Gemini AI for analysis")
            scored_clips = extractor.analyze_with_gemini(transcript_segments, gemini_key)
            
            # Normalize AI response
            for clip in scored_clips:
                clip['title'] = clip.get('reason', 'AI Selected Clip')
                clip.setdefault('start', 0)
                clip.setdefault('end', clip['start'] + clip_length)
                clip.setdefault('viral_score', 80)
        else:
            # 3. Generate candidates
            candidates = extractor.generate_candidates(video_info, clip_length, window_step)
            
            # 4. Score candidates
            scored_clips = []
            logger.info(f"Scoring {len(candidates)} candidates...")
            for i, cand in enumerate(candidates):
                score_data = extractor.score_clip(url, cand['start'], cand['end'], transcript_segments)
                score_data['title'] = cand.get('title', f'Clip {i+1}')
                scored_clips.append(score_data)
            
            # Sort by viral score
            scored_clips.sort(key=lambda x: x['viral_score'], reverse=True)
        
        # Select top non-overlapping clips
        final_clips = []
        used_ranges = []
        
        for clip in scored_clips:
            overlaps = False
            for used in used_ranges:
                if not (clip['end'] <= used['start'] or clip['start'] >= used['end']):
                    overlaps = True
                    break
            
            if not overlaps:
                final_clips.append(clip)
                used_ranges.append(clip)
                
                if len(final_clips) >= num_clips:
                    break
        
        # Add formatted times and YouTube URLs and Download Links
        import urllib.parse
        encoded_url = urllib.parse.quote(url)
        
        for clip in final_clips:
            clip['start'] = float(clip['start'])
            clip['end'] = float(clip['end'])
            clip['start_formatted'] = seconds_to_time(clip['start'])
            clip['end_formatted'] = seconds_to_time(clip['end'])
            clip['youtube_url'] = f"https://youtube.com/watch?v={video_id}&t={int(clip['start'])}"
            clip['download_link'] = f"/download_file?url={encoded_url}&start={clip['start']}&end={clip['end']}&quality={quality}"
        
        return jsonify({
            "success": True,
            "video_id": video_id,
            "video_title": video_info.get('title'),
            "video_duration": video_info.get('duration'),
            "clips": final_clips
        })
        
    except Exception as e:
        logger.error(f"Error in /analyze: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/download_file', methods=['GET'])
def download_file_endpoint():
    """Download the actual clip file"""
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({"error": "Missing 'url' parameter"}), 400
        
        start = float(request.args.get('start', 0))
        end = float(request.args.get('end', 0))
        quality = request.args.get('quality', '720')
        
        if end <= start:
             return jsonify({"error": "Invalid time range"}), 400
        
        extractor = ViralClipExtractor()
        file_path = extractor.download_clip(url, start, end, quality)
        
        if file_path and os.path.exists(file_path):
            @after_this_request
            def remove_file(response):
                try:
                   pass # windows file locking might prevent removal if not closed, but send_file should handle it
                   # os.remove(file_path) # Defer removal or use a temp dir cleaner properly
                except Exception as error:
                    app.logger.error("Error removing file", error)
                return response
            
            return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path))
        else:
            return jsonify({"error": "Failed to download clip"}), 500
            
    except Exception as e:
        logger.error(f"Error in /download: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/docs', methods=['GET'])
def docs():
    """API Documentation & Credits"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Viral Clip Extractor API | Documentation</title>
        <style>
            :root { --primary: #3b82f6; --secondary: #8b5cf6; --bg: #0f172a; --text: #f8fafc; --card-bg: #1e293b; }
            body { 
                font-family: 'Segoe UI', system-ui, sans-serif; 
                line-height: 1.6; 
                color: var(--text); 
                background: var(--bg); 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 40px 20px;
                opacity: 0;
                animation: fadeIn 0.8s ease-out forwards;
            }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
            @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

            h1 { font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            h2 { margin-top: 3rem; border-bottom: 2px solid #334155; padding-bottom: 0.5rem; color: #94a3b8; }
            h3 { color: #60a5fa; margin-top: 0; }
            
            .badge { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.9rem; vertical-align: middle; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
            
            .card { 
                background: var(--card-bg); 
                padding: 25px; 
                border-radius: 16px; 
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); 
                margin-bottom: 25px; 
                border: 1px solid #334155; 
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover { transform: translateY(-5px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.4); border-color: #60a5fa; }
            
            code { font-family: 'Consolas', monospace; background: #0f172a; padding: 4px 8px; border-radius: 6px; color: #f472b6; border: 1px solid #334155; }
            pre { background: #020617; color: #e2e8f0; padding: 20px; border-radius: 12px; overflow-x: auto; font-size: 0.9rem; border: 1px solid #334155; }
            
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #334155; }
            th { color: #94a3b8; font-weight: 600; }
            
            .method { font-weight: bold; color: #34d399; }
            .url { color: #cbd5e1; }
            
            .warning { background: rgba(249, 115, 22, 0.1); border-left: 4px solid #f97316; padding: 15px; margin: 20px 0; border-radius: 0 8px 8px 0; }
            .tip { background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 15px; margin: 20px 0; border-radius: 0 8px 8px 0; }
            
            .creator-highlight {
                background: linear-gradient(270deg, #ff00cc, #333399, #60a5fa);
                background-size: 600% 600%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
                font-size: 1.5em;
                animation: gradient 3s ease infinite;
                display: inline-block;
                padding: 0 5px;
            }
            
            footer { margin-top: 80px; text-align: center; color: #64748b; font-size: 1.1rem; border-top: 1px solid #334155; padding-top: 40px; }
        </style>
    </head>
    <body>
        <h1>Viral Clip Extractor API <span class="badge">v1.3</span></h1>
        <p class="lead">Transform long-form YouTube videos into engaging viral clips using sophisticated AI analysis.</p>
        
        <div class="warning">
            <strong>🔐 Security Best Practice:</strong> Do not pass API keys in URLs for public applications. 
            Configure <code>GEMINI_API_KEY</code> and <code>NVIDIA_API_KEY</code> as Environment Variables on your server.
        </div>

        <h2>🧠 Analysis Modes & Usage</h2>
        
        <div class="card">
            <h3>1. Heuristic Mode (Non-AI)</h3>
            <p><strong>Best for:</strong> High-energy content, gaming, reactions. Fast and free.</p>
            <div class="tip">No API Key required.</div>
            <pre>GET /clips?url={youtube_url}&mode=heuristic&num=5</pre>
        </div>

        <div class="card">
            <h3>2. Gemini AI Mode</h3>
            <p><strong>Best for:</strong> Podcasts, storytelling, general dialogue. Balanced performance.</p>
            <div class="warning">Requires <code>GEMINI_API_KEY</code> environment variable (or param).</div>
            <pre>GET /clips?url={youtube_url}&mode=ai&num=5</pre>
        </div>

        <div class="card">
            <h3>3. Nvidia/DeepSeek AI Mode</h3>
            <p><strong>Best for:</strong> Technical content, debates, complex reasoning.</p>
            <div class="warning">Requires <code>NVIDIA_API_KEY</code> environment variable (or param).</div>
            <pre>GET /clips?url={youtube_url}&mode=nvidia&num=5</pre>
        </div>

        <h2>📚 Endpoints</h2>

        <div class="card">
            <h3>GET /clips</h3>
            <p>Main endpoint to analyze and get clip suggestions.</p>
            
            <h4>Parameters</h4>
            <table>
                <tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr>
                <tr><td><code>url</code></td><td>string</td><td>Yes</td><td>YouTube Video URL</td></tr>
                <tr><td><code>mode</code></td><td>string</td><td>No</td><td><code>heuristic</code> (default), <code>ai</code>, <code>nvidia</code></td></tr>
                <tr><td><code>num</code></td><td>int</td><td>No</td><td>Number of clips (default: 5)</td></tr>
                <tr><td><code>quality</code></td><td>string</td><td>No</td><td>Video height (e.g., <code>720</code>, <code>1080</code>)</td></tr>
            </table>

            <h4>Example Response</h4>
            <pre>{
  "success": true,
  "video_title": "Podcast Episode 1",
  "mode": "ai",
  "clips": [
    {
      "title": "Shocking Reveal",
      "start": 120.5,
      "end": 150.0,
      "viral_score": 95,
      "download_link": "/download_file?url=...&start=120.5&end=150"
    }
  ]
}</pre>
        </div>

        <div class="card">
            <h3>GET /download_file</h3>
            <p>Download a specific clip directly.</p>
            <pre>GET /download_file?url={url}&start={start}&end={end}</pre>
        </div>

        <footer>
            <p>Developed with ❤️ by <span class="creator-highlight">yash@dev</span></p>
            <p>&copy; 2026 Viral Clip Extractor API</p>
        </footer>
    </body>
    </html>
    """
    return html

# Local development and Production
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

