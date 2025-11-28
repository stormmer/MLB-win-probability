# Quick Deployment Guide

## Fastest Option: Render (5 minutes)

1. **Sign up** at [render.com](https://render.com) (free)

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Or use "Public Git repository" and paste your repo URL

3. **Configure**:
   - **Name**: `mlb-win-probability-dashboard` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --config gunicorn_config.py dashboard.wsgi:server`
   - **Plan**: Free (or upgrade for better performance)

4. **Deploy**:
   - Click "Create Web Service"
   - Wait 5-10 minutes for first deployment
   - Your dashboard will be live at `https://your-app-name.onrender.com`

**Note**: Free tier may spin down after 15 minutes of inactivity. First request after spin-down takes ~30 seconds.

---

## Alternative: Railway (Also 5 minutes)

1. **Sign up** at [railway.app](https://railway.app) (free)

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Auto-deploy**:
   - Railway auto-detects Python
   - Uses `railway.json` configuration
   - Deploys automatically

4. **Done**:
   - Your dashboard at `https://your-app-name.up.railway.app`

---

## Before Deploying

Make sure you have:
- ✅ All files committed to Git
- ✅ `requirements.txt` includes all dependencies
- ✅ Data files and model are in the repository (or uploaded separately)
- ✅ Team logos in `assets/logos/`
- ✅ MLB logo in `assets/mlb_logo.png`

---

## Testing Locally First

Test the production setup:

```bash
pip install gunicorn
gunicorn --config gunicorn_config.py dashboard.wsgi:server
```

Visit `http://localhost:8050` - if it works, deployment should work too!

---

## Troubleshooting

**Dashboard shows error**: Check logs on your platform
**Model not found**: Ensure `models/` folder is in repository
**Logos missing**: Ensure `assets/` folder is in repository
**Port error**: Platform will set PORT automatically - don't hardcode

---

For detailed instructions, see `DEPLOYMENT.md`

