# GitHub Deployment Instructions

Your repository is ready! Follow these steps to push to GitHub:

## Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in:
   - **Repository name**: `mlb-win-probability-dashboard` (or your choice)
   - **Description**: "MLB Win Probability Prediction Dashboard"
   - **Visibility**: Public (or Private if you prefer)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Push Your Code

After creating the repository, GitHub will show you commands. Use these:

```bash
cd "C:\Users\abson\OneDrive\Documents\MLB win probability"
git remote add origin https://github.com/YOUR_USERNAME/mlb-win-probability-dashboard.git
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME` with your GitHub username!**

## Step 3: Deploy to Render (Recommended)

Once your code is on GitHub:

1. Go to [render.com](https://render.com) and sign up/login
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect GitHub"** and authorize Render
4. Select your repository: `mlb-win-probability-dashboard`
5. Configure:
   - **Name**: `mlb-win-probability-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --config gunicorn_config.py dashboard.wsgi:server`
   - **Plan**: Free (or upgrade)
6. Click **"Create Web Service"**
7. Wait 5-10 minutes for deployment
8. Your dashboard will be live at: `https://mlb-win-probability-dashboard.onrender.com`

## Alternative: Deploy to Railway

1. Go to [railway.app](https://railway.app) and sign up/login
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway will auto-detect and deploy!
6. Your dashboard will be live at: `https://your-app.up.railway.app`

## Important Notes

⚠️ **Large Files**: If your `data/` or `models/` folders are very large (>100MB), you may need to:
- Use Git LFS (Large File Storage)
- Or upload them separately after deployment
- Or use cloud storage (S3, etc.) and modify code to load from there

✅ **Everything is ready**: All deployment files are committed and ready to go!

---

**Need help?** Check `DEPLOYMENT.md` for detailed instructions.

