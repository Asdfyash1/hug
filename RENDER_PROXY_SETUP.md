# Proxy Management Guide - Viral Clip Extractor

## Overview

The Viral Clip Extractor now includes advanced proxy management to bypass YouTube's datacenter IP blocking. This system supports:

- ✅ **Single Proxy** - Add individual proxies manually
- ✅ **WebShare API Integration** - Load entire proxy lists automatically
- ✅ **Automatic Rotation** - Round-robin proxy selection
- ✅ **Bandwidth Tracking** - Monitor usage per proxy and total
- ✅ **Web UI** - Beautiful interface at `/proxy` endpoint

---

## Quick Start

### Option 1: Environment Variables (Recommended for Render)

**Single Proxy:**
```bash
PROXY_URL=31.59.20.176:6754:nntlrciu:sx2noxvkj6y7
```

**WebShare API:**
```bash
PROXY_API_URL=https://proxy.webshare.io/api/v2/proxy/list/download/ihkhaegfvfchhpyzyxujdntbqmuoushqwvczhztu/-/any/username/direct/-/?plan_id=12753876
```

### Option 2: Web UI (Recommended for Manual Management)

1. Navigate to: `https://your-app.onrender.com/proxy`
2. Add proxies using the web interface
3. Monitor bandwidth usage in real-time

---

## Proxy Formats Supported

### Format 1: `ip:port:user:pass`
```
31.59.20.176:6754:nntlrciu:sx2noxvkj6y7
```

### Format 2: Standard HTTP Proxy
```
http://nntlrciu:sx2noxvkj6y7@31.59.20.176:6754
```

### Format 3: WebShare API URL
```
https://proxy.webshare.io/api/v2/proxy/list/download/YOUR_TOKEN/-/any/username/direct/-/?plan_id=YOUR_PLAN_ID
```

---

## Render Deployment Setup

### Step 1: Go to Render Dashboard
1. Visit: https://dashboard.render.com/
2. Select your service: `kimi-agent-viral-clip-extractor`
3. Click **Environment** tab

### Step 2: Add Environment Variable

**For Single Proxy:**
- **Key:** `PROXY_URL`
- **Value:** `31.59.20.176:6754:nntlrciu:sx2noxvkj6y7`

**For WebShare API (Recommended):**
- **Key:** `PROXY_API_URL`
- **Value:** Your WebShare API URL

### Step 3: Save and Deploy
- Click **Save Changes**
- Render will automatically redeploy (~2-3 minutes)

---

## Web UI Features

### Access the UI
```
https://your-app.onrender.com/proxy
```

### Dashboard Stats
- **Total Proxies** - Number of active proxies in rotation
- **Bandwidth Used** - Total MB consumed
- **Remaining** - Available bandwidth (out of 1GB limit)
- **Current Index** - Next proxy in rotation queue

### Add Proxies
1. **Single Proxy**
   - Enter: `ip:port:user:pass`
   - Click "Add Single Proxy"

2. **WebShare API**
   - Paste your API URL
   - Click "Load from API"
   - All proxies load automatically

### Monitor Usage
- View bandwidth per proxy
- Track total usage
- Visual progress bar for monthly limit

---

## API Endpoints

### GET `/proxy`
Returns web UI for proxy management

**Response:** HTML page with dashboard

### POST `/proxy`
Update proxy configuration

**Add Single Proxy:**
```json
{
  "action": "add_single",
  "proxy": "31.59.20.176:6754:nntlrciu:sx2noxvkj6y7"
}
```

**Load from API:**
```json
{
  "action": "add_api",
  "api_url": "https://proxy.webshare.io/api/v2/proxy/list/download/..."
}
```

**Get Stats:**
```json
{
  "action": "get_stats"
}
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_proxies": 10,
    "total_bandwidth_mb": 45.2,
    "bandwidth_remaining_mb": 978.8,
    "per_proxy_stats": [...]
  }
}
```

---

## Bandwidth Tracking

### How It Works
- Every API request tracks bandwidth used
- Estimates based on video metadata and transcript size
- Persists to `proxy_stats.json` file
- Survives server restarts

### Bandwidth Estimates
- **Video Info:** ~10-50 KB
- **Transcript:** ~20-100 KB
- **Clip Download (30s, 720p):** ~5-15 MB
- **Full Request (3 clips):** ~15-50 MB

### Monthly Limit
- **Total:** 1 GB (1024 MB)
- **Estimated Requests:** ~20-50 full clip extractions
- **Monitor:** Check `/proxy` dashboard regularly

---

## Proxy Rotation

### How It Works
1. ProxyManager maintains a list of proxies
2. Each request uses the next proxy (round-robin)
3. Distributes load evenly across all proxies
4. Prevents single proxy from being rate-limited

### Example
```
Request 1 → Proxy A
Request 2 → Proxy B
Request 3 → Proxy C
Request 4 → Proxy A (rotation restarts)
```

---

## Your Proxy List

You provided 10 proxies. Here's how to use them:

### Individual Proxies
```
31.59.20.176:6754:nntlrciu:sx2noxvkj6y7
23.95.150.145:6114:nntlrciu:sx2noxvkj6y7
198.23.239.134:6540:nntlrciu:sx2noxvkj6y7
45.38.107.97:6014:nntlrciu:sx2noxvkj6y7
107.172.163.27:6543:nntlrciu:sx2noxvkj6y7
198.105.121.200:6462:nntlrciu:sx2noxvkj6y7
64.137.96.74:6641:nntlrciu:sx2noxvkj6y7
216.10.27.159:6837:nntlrciu:sx2noxvkj6y7
23.26.71.145:5628:nntlrciu:sx2noxvkj6y7
23.229.19.94:8689:nntlrciu:sx2noxvkj6y7
```

### WebShare API URL
```
https://proxy.webshare.io/api/v2/proxy/list/download/ihkhaegfvfchhpyzyxujdntbqmuoushqwvczhztu/-/any/username/direct/-/?plan_id=12753876
```

**Recommended:** Use the API URL in Render for automatic updates

---

## Testing

### Local Test
```bash
# Set environment variable
export PROXY_URL="31.59.20.176:6754:nntlrciu:sx2noxvkj6y7"

# Run test
python test_proxy_minimal.py
```

### Production Test
```bash
curl "https://your-app.onrender.com/clips?url=https://youtu.be/uVkFrqugXFQ&mode=nvidia&nvidia_key=YOUR_KEY&num=3"
```

### Check Bandwidth
Visit: `https://your-app.onrender.com/proxy`

---

## Troubleshooting

### Proxy Not Working
1. Check format: `ip:port:user:pass`
2. Verify credentials are correct
3. Test proxy manually with curl
4. Check `/proxy` dashboard for errors

### Bandwidth Exceeded
1. Visit `/proxy` to check usage
2. Wait for monthly reset
3. Consider upgrading proxy plan
4. Reduce number of clips per request

### Rotation Not Working
1. Ensure multiple proxies are added
2. Check `/proxy` dashboard
3. Verify `current_proxy_index` changes
4. Review logs for errors

---

## Best Practices

1. **Use WebShare API URL** - Automatic updates, easier management
2. **Monitor Bandwidth** - Check `/proxy` dashboard regularly
3. **Start Small** - Test with 1-2 requests before scaling
4. **Rotate Proxies** - Add multiple proxies for better distribution
5. **Track Usage** - Keep bandwidth under 1GB/month limit

---

## Support

For issues or questions:
- Check `/proxy` dashboard for real-time stats
- Review `proxy_stats.json` for detailed logs
- Test locally before deploying to production
