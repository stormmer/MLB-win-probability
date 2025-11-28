# Streamlit Deployment Guide

## Option 1: Streamlit Cloud (Easiest - Recommended)

### Step 1: Push to GitHub
Your code is already on GitHub at: `https://github.com/stormmer/MLB-win-probability`

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the form**:
   - **Repository**: `stormmer/MLB-win-probability`
   - **Branch**: `main`
   - **Main file path**: `dashboard/streamlit_app.py`
   - **App URL**: `mlb-win-probability` (or your choice)
5. **Click "Deploy"**
6. **Wait 2-3 minutes** for deployment
7. **Your app will be live at**: `https://mlb-win-probability.streamlit.app`

That's it! Streamlit Cloud handles everything automatically.

---

## Option 2: Railway (Alternative)

1. **Go to Railway** → New Project → Deploy from GitHub
2. **Select your repository**
3. **In Settings → Deploy**, set:
   - **Start Command**: `streamlit run dashboard/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. **Add Environment Variable**:
   - `PORT` (Railway sets this automatically)
5. **Deploy**

---

## Option 3: Render

1. **Go to Render** → New Web Service
2. **Connect GitHub** → Select repository
3. **Configure**:
   - **Build Command**: `pip install -r streamlit_requirements.txt`
   - **Start Command**: `streamlit run dashboard/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. **Deploy**

---

## Option 4: Docker

Create a `Dockerfile.streamlit`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY streamlit_requirements.txt .
RUN pip install --no-cache-dir -r streamlit_requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then deploy to any Docker hosting platform.

---

## Quick Deploy: Streamlit Cloud (5 minutes)

**Recommended method** - Just go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo!

Your dashboard will be live in minutes with:
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Auto-deploy on git push
- ✅ No configuration needed
- ✅ Professional URL

---

## Files Needed

Make sure these are in your repository:
- ✅ `dashboard/streamlit_app.py`
- ✅ `requirements.txt` (or `streamlit_requirements.txt`)
- ✅ `src/mlb_win_pred/` (all source files)
- ✅ `data/processed/` (if you want data)
- ✅ `models/` (if you want predictions)
- ✅ `assets/` (logos)

---

## Troubleshooting

**App not loading**: Check Streamlit Cloud logs
**Import errors**: Make sure `src/` directory is in repository
**Missing files**: App will show warnings but still work

Streamlit Cloud is the easiest option - just connect GitHub and deploy!

