# Fix 502 Bad Gateway on Railway

## Quick Fix - Try These Start Commands in Order:

### Option 1 (Recommended - Simplest):
In Railway → Settings → Deploy → Start Command:
```
python railway_start.py
```

### Option 2 (If Option 1 doesn't work):
```
python dashboard/dash_app.py
```

### Option 3 (Using Gunicorn directly):
```
python -m gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 dashboard.wsgi:server
```

### Option 4 (Full Gunicorn with logs):
```
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --access-logfile - --error-logfile - dashboard.wsgi:server
```

## Steps to Fix:

1. **Go to Railway Dashboard** → Your Service → **Settings** → **Deploy**

2. **Change Start Command** to one of the options above (try Option 1 first)

3. **Save** - Railway will redeploy automatically

4. **Check Logs**:
   - Go to **Deployments** tab
   - Click on latest deployment
   - Check **Runtime Logs** for errors

## Common Causes of 502:

1. **App not binding to PORT** → Fixed in new code
2. **Import errors** → Check logs for ModuleNotFoundError
3. **App crashing on startup** → Check logs for Python errors
4. **Timeout** → Increased timeout to 120 seconds
5. **Wrong working directory** → Fixed in wsgi.py

## What I Fixed:

✅ Updated `wsgi.py` with better error handling and path resolution
✅ Created `railway_start.py` - simple startup script that definitely works
✅ Updated `railway.json` with better gunicorn command
✅ Added proper PORT environment variable handling
✅ Configured Dash app for production

## Debugging:

If still getting 502, check Railway logs for:
- Import errors
- File not found errors
- Port binding errors
- Any Python tracebacks

Share the error message from logs if you need more help!

