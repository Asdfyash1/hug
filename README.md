# 🎬 Viral Clip Extractor API

Turn long-form YouTube videos into engaging, viral-ready clips for TikTok, Shorts, and Reels using AI.

**Live Demo / Docs:** `https://your-app-url.onrender.com/docs`

## ✨ Features

- **🧠 Three Powerful Modes:**
  1.  **Heuristic (Free/Fast):** Uses keyword analysis and audio/visual cues. Great for high-energy content.
  2.  **Gemini AI (Balanced):** Uses Google's Gemini 2.0 Flash for context-aware clipping.
  3.  **Nvidia/DeepSeek AI (Advanced):** Uses DeepSeek V3/R1 via Nvidia API for deep reasoning and technical content.
- **🚀 High Performance:** Optimized `yt-dlp` and `ffmpeg` integration for fast processing.
- **☁️ Cloud Ready:** Dockerized and configured for [Render](https://render.com) deployment.
- **🔒 Secure:** Environment variable support for API keys.
- **📄 Auto-Documentation:** Built-in interactive API docs at `/docs`.

## 🛠️ Installation

### Option 1: Docker (Recommended)
```bash
docker build -t viral-clips .
docker run -p 5000:5000 --env-file .env viral-clips
```

### Option 2: Local Python
1.  **Install FFmpeg:** Ensure `ffmpeg` is in your system PATH.
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run API:**
    ```bash
    python api/index.py
    ```

## 🔑 Configuration

Create a `.env` file or set these environment variables in your cloud dashboard:

| Variable | Description | Required For |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API Key | `mode=ai` |
| `NVIDIA_API_KEY` | Nvidia/DeepSeek API Key | `mode=nvidia` |
| `PROXY_URL` | Single proxy (format: `ip:port:user:pass`) | YouTube access (optional) |
| `PROXY_API_URL` | WebShare API URL for proxy list | YouTube access (recommended) |
| `PORT` | Server Port (Default: 5000) | Deployment |

### 🔒 Proxy Setup (Bypass YouTube Bot Detection)

For reliable YouTube access, especially on cloud platforms, configure proxies:

**Option 1: Single Proxy**
```bash
PROXY_URL=31.59.20.176:6754:username:password
```

**Option 2: WebShare API (Recommended)**
```bash
PROXY_API_URL=https://proxy.webshare.io/api/v2/proxy/list/download/YOUR_TOKEN/-/any/username/direct/-/
```

**Manage Proxies:** Visit `/proxy` endpoint for web UI to add/monitor proxies and track bandwidth usage.

## 📚 API Usage

### Get Clips
**Endpoint:** `GET /clips`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `url` | string | **Required** | YouTube Video URL |
| `mode` | string | `heuristic` | `heuristic`, `ai`, or `nvidia` |
| `num` | int | `5` | Number of clips to generate |

**Example:**
```bash
curl "http://localhost:5000/clips?url=https://youtu.be/VIDEO_ID&mode=ai"
```

### Download Clip
**Endpoint:** `GET /download_file`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `url` | string | Original YouTube URL |
| `start` | float | Start time (seconds) |
| `end` | float | End time (seconds) |

### Proxy Management
**Endpoint:** `GET /proxy` (Web UI) or `POST /proxy` (API)

**Web UI Features:**
- View bandwidth usage and statistics
- Add single proxy or load from WebShare API
- Monitor proxy rotation and performance

**API Actions:**
```bash
# Add single proxy
curl -X POST http://localhost:5000/proxy \
  -H "Content-Type: application/json" \
  -d '{"action": "add_single", "proxy": "ip:port:user:pass"}'

# Load from WebShare API
curl -X POST http://localhost:5000/proxy \
  -H "Content-Type: application/json" \
  -d '{"action": "add_api", "api_url": "https://proxy.webshare.io/..."}'

# Get stats
curl -X POST http://localhost:5000/proxy \
  -H "Content-Type: application/json" \
  -d '{"action": "get_stats"}'
```

## ⚡ Performance Optimizations

- **Default Quality:** 480p (50% faster downloads, use `&quality=720` for higher quality)
- **Concurrent Downloads:** 4 parallel fragment downloads
- **AI Processing:** Max 10 candidates with 30s window step (~60% faster)
- **Proxy Support:** Residential proxies for YouTube bot detection bypass
- **Smart Caching:** Bandwidth tracking and persistent stats

## 📝 Credits
Developed with ❤️ by **yash@dev**

Powered by:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg](https://ffmpeg.org/)
- [Google Gemini](https://ai.google.dev/)
- [Nvidia DeepSeek](https://build.nvidia.com/)
