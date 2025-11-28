# Railway Deployment Fix

If your Railway deployment is failing, try these solutions:

## Solution 1: Update Start Command in Railway Dashboard

1. Go to your Railway project dashboard
2. Click on your service
3. Go to **Settings** → **Deploy**
4. Change the **Start Command** to one of these:

**Option A (Recommended):**
```
cd dashboard && python -m gunicorn --config ../gunicorn_config.py wsgi:server
```

**Option B (Alternative):**
```
python start.py
```

**Option C (Simple):**
```
cd dashboard && gunicorn --bind 0.0.0.0:$PORT wsgi:server
```

5. Click **Save**
6. Railway will redeploy automatically

## Solution 2: Check Build Logs

1. In Railway dashboard, click on your service
2. Go to **Deployments** tab
3. Click on the latest deployment
4. Check **Build Logs** for errors

Common issues:
- Missing dependencies → Check `requirements.txt` includes everything
- Path errors → The start command might need adjustment
- Port issues → Railway sets PORT automatically

## Solution 3: Use Environment Variables

In Railway dashboard → Settings → Variables, add:
- `PORT` (Railway sets this automatically, but you can verify)
- `PYTHON_VERSION=3.11`

## Solution 4: Simplified Start Command

If gunicorn is causing issues, try this simpler command:

```
python dashboard/dash_app.py
```

Then update `dash_app.py` to use the PORT environment variable (already done).

## Solution 5: Check File Structure

Make sure your repository has:
- ✅ `requirements.txt` (with gunicorn)
- ✅ `dashboard/wsgi.py`
- ✅ `dashboard/dash_app.py`
- ✅ `gunicorn_config.py`
- ✅ `railway.json`

## Still Not Working?

1. **Check Railway logs**: Go to your service → Logs tab
2. **Common errors**:
   - `ModuleNotFoundError` → Add missing package to requirements.txt
   - `FileNotFoundError` → Check file paths are correct
   - `Port already in use` → Railway handles this automatically
3. **Try the simple start command**: `python dashboard/dash_app.py`

## Quick Test Locally

Test if it works locally first:
```bash
pip install gunicorn
cd dashboard
gunicorn --bind 0.0.0.0:8050 wsgi:server
```

If this works locally, it should work on Railway!

