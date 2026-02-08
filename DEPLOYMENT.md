# 🚀 Deployment Guide

Complete guide to deploy the Viral Clip Extractor API to Vercel and GitHub.

## Quick Deploy (Recommended)

### Option 1: Vercel Dashboard (Easiest)

1. **Push to GitHub**
   ```bash
   # Create a new repository on GitHub first
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/viral-clip-extractor-api.git
   git push -u origin main
   ```

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Sign up/login with GitHub
   - Click "New Project"
   - Import your repository
   - Vercel auto-detects Python
   - Click "Deploy"

3. **Done!** Your API is live at `https://your-project.vercel.app`

---

### Option 2: Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   cd viral-clip-extractor-api
   vercel --prod
   ```

4. **Follow prompts** - Vercel will detect Python automatically

---

## GitHub + Vercel Integration (Auto-Deploy)

### Setup GitHub Repository

1. **Create repository on GitHub**
   - Go to github.com/new
   - Name: `viral-clip-extractor-api`
   - Make it Public or Private
   - Don't initialize with README

2. **Push local code**
   ```bash
   cd viral-okcomputer/output/viral-clip-api
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/viral-clip-extractor-api.git
   git push -u origin main
   ```

### Setup Vercel Project

1. **Import from GitHub**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Select your repository
   - Framework Preset: `Other`
   - Root Directory: `./`
   - Build Command: (leave empty for Python)
   - Output Directory: (leave empty)

2. **Environment Variables** (if needed)
   - Click "Environment Variables"
   - Add any required variables
   - Click "Deploy"

3. **Auto-Deploy Enabled**
   - Every push to `main` branch auto-deploys
   - Pull requests get preview deployments

---

## GitHub Actions CI/CD

The repository includes a GitHub Actions workflow for automated deployment.

### Setup Secrets

1. **Get Vercel Tokens**
   ```bash
   vercel login
   vercel tokens create
   ```

2. **Get Project ID**
   ```bash
   cd your-project
   vercel
   # Check .vercel/project.json
   ```

3. **Add to GitHub Secrets**
   - Go to Repository → Settings → Secrets → Actions
   - Add:
     - `VERCEL_TOKEN` - Your token
     - `VERCEL_ORG_ID` - From `.vercel/project.json`
     - `VERCEL_PROJECT_ID` - From `.vercel/project.json`

### Workflow Features

- ✅ Runs tests on every PR
- ✅ Auto-deploys on merge to main
- ✅ Shows deployment status in PRs

---

## Configuration

### Vercel Settings

Edit `vercel.json` to customize:

```json
{
  "version": 2,
  "name": "viral-clip-extractor-api",
  "functions": {
    "api/index.py": {
      "maxDuration": 60    // Max 60 seconds (Vercel limit)
    }
  }
}
```

### Python Runtime

Vercel uses Python 3.9 by default. To change:

```json
{
  "builds": [{
    "src": "api/index.py",
    "use": "@vercel/python",
    "config": {
      "runtime": "python3.9"
    }
  }]
}
```

### Custom Domain

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your domain
3. Follow DNS configuration instructions

---

## Testing Your Deployment

### 1. Health Check
```bash
curl https://your-project.vercel.app/health
```

Expected:
```json
{"status": "healthy"}
```

### 2. API Info
```bash
curl https://your-project.vercel.app/
```

### 3. Test Analysis
```bash
curl -X POST https://your-project.vercel.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=dQw4w9WgXcQ", "num_clips": 3}'
```

---

## Troubleshooting

### Build Failures

**Problem**: Build fails with dependency errors

**Solution**:
```bash
# Update requirements.txt
pip freeze > requirements.txt

# Or manually specify versions
flask==3.0.3
yt-dlp==2024.8.6
```

### Timeout Issues

**Problem**: 504 Gateway Timeout

**Solution**:
- Reduce `clip_length` or `num_clips` in requests
- Videos > 2 hours may timeout
- Consider splitting long videos

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
- Ensure all dependencies in `requirements.txt`
- Check Python version compatibility
- Verify file structure matches `vercel.json`

### CORS Errors

**Problem**: Frontend can't access API

**Solution**:
- CORS is enabled by default in `api/index.py`
- Check `flask-cors` is in requirements.txt

---

## Monitoring

### Vercel Analytics

1. Go to Vercel Dashboard → Your Project → Analytics
2. View:
   - Request count
   - Response times
   - Error rates
   - Bandwidth usage

### Logs

```bash
# View live logs
vercel logs your-project --tail
```

Or in Dashboard → Project → Logs

---

## Updating Your API

### Method 1: Git Push (Auto-Deploy)

```bash
# Make changes
git add .
git commit -m "Update feature"
git push origin main
# Vercel auto-deploys!
```

### Method 2: Vercel CLI

```bash
vercel --prod
```

---

## Environment Variables

### Add to Vercel

1. Dashboard → Project → Settings → Environment Variables
2. Or via CLI:
   ```bash
   vercel env add VARIABLE_NAME
   ```

### Use in Code

```python
import os

api_key = os.environ.get('API_KEY')
```

---

## Security Best Practices

1. **Never commit `.env` files**
   - Already in `.gitignore`

2. **Use environment variables for secrets**
   - API keys
   - Database URLs
   - Private tokens

3. **Enable Vercel Authentication**
   - Dashboard → Project → Settings → Security

4. **Rate Limiting**
   - Consider adding Flask-Limiter for production

---

## Next Steps

- [ ] Add custom domain
- [ ] Set up monitoring/alerts
- [ ] Add rate limiting
- [ ] Implement caching
- [ ] Add more languages support
- [ ] Create mobile app

---

## Support

- **Vercel Docs**: [vercel.com/docs](https://vercel.com/docs)
- **Python Runtime**: [vercel.com/docs/functions/serverless-functions/runtimes/python](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- **yt-dlp Docs**: [github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp)

---

**Happy Deploying! 🚀**
