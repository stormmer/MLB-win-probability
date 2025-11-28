# Railway Deployment Debugging Guide

## Current Issue: Application Failed to Respond

This means the app is crashing on startup. Here's how to fix it:

## Step 1: Check Railway Logs

1. Go to Railway Dashboard → Your Service
2. Click **"View Logs"** or go to **Deployments** → Latest → **Runtime Logs**
3. Look for Python errors, import errors, or file not found errors
4. **Copy the full error message** - this will help diagnose the issue

## Step 2: Try These Start Commands (in order)

### Option 1: Simple Python Script (Recommended)
```
python railway_start.py
```
This will show detailed startup logs.

### Option 2: Direct Dash App
```
python dashboard/dash_app.py
```

### Option 3: Gunicorn with Single Worker
```
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level debug dashboard.wsgi:server
```

## Step 3: Common Issues & Fixes

### Issue 1: Missing Data Files
**Error**: `FileNotFoundError: games_processed.csv not found`

**Fix**: The app now handles missing files gracefully, but you should:
- Make sure `data/processed/games_processed.csv` is in your repository
- Or the app will show a warning but still start

### Issue 2: Missing Model File
**Error**: `FileNotFoundError: win_model_xgb.pkl not found`

**Fix**: 
- Make sure `models/win_model_xgb.pkl` is in your repository
- Or the app will show a warning but still start (predictions won't work)

### Issue 3: Import Errors
**Error**: `ModuleNotFoundError: No module named 'mlb_win_pred'`

**Fix**: 
- Check that `src/mlb_win_pred/` directory is in repository
- Check that all files in `requirements.txt` are installed

### Issue 4: Port Binding
**Error**: `Address already in use` or port errors

**Fix**: 
- Railway sets `$PORT` automatically
- The code now uses `$PORT` correctly
- Make sure start command uses `$PORT` not hardcoded `8050`

## Step 4: Verify Files Are in Repository

Run locally:
```bash
git ls-files | grep -E "(data|models|src)"
```

Make sure you see:
- `src/mlb_win_pred/` files
- `data/processed/games_processed.csv` (if you want it)
- `models/win_model_xgb.pkl` (if you want it)

## Step 5: Test Locally First

Before deploying, test the production setup locally:

```bash
# Install gunicorn
pip install gunicorn

# Test with railway_start.py
python railway_start.py

# Or test with gunicorn
gunicorn --bind 0.0.0.0:8050 --workers 1 dashboard.wsgi:server
```

If it works locally, it should work on Railway.

## Step 6: What I Just Fixed

✅ Added error handling to prevent crashes
✅ Made app work even if data/model files are missing
✅ Added detailed logging in `railway_start.py`
✅ Improved import error handling
✅ Made `load_data_and_model()` more robust

## Still Not Working?

1. **Share the error logs** from Railway
2. **Check if files are committed** to Git
3. **Try the simple start command**: `python railway_start.py`
4. **Check Railway logs** for the exact error message

The app should now start even if some files are missing (it will show warnings but won't crash).

