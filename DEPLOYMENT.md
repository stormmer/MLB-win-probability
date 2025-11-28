# MLB Win Probability Dashboard - Deployment Guide

This guide covers multiple deployment options for the MLB Win Probability Dashboard.

## Prerequisites

- Python 3.11+
- All dependencies installed (`pip install -r requirements.txt`)
- Data files and trained model in place
- Git repository (for most platforms)

## Quick Deploy Options

### Option 1: Render (Recommended - Easiest)

1. **Create a Render account** at [render.com](https://render.com)

2. **Create a new Web Service**:
   - Connect your GitHub repository
   - Select "Web Service"
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn --config gunicorn_config.py dashboard.wsgi:server`
     - **Environment**: Python 3

3. **Add Environment Variables** (if needed):
   - `PYTHON_VERSION=3.11.0`

4. **Deploy**: Click "Create Web Service"

The dashboard will be available at `https://your-app-name.onrender.com`

**Note**: Free tier may spin down after inactivity. Upgrade to paid plan for always-on service.

---

### Option 2: Railway

1. **Create a Railway account** at [railway.app](https://railway.app)

2. **Create a new project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure**:
   - Railway will auto-detect Python
   - Uses `railway.json` for configuration
   - Start command: `gunicorn --config gunicorn_config.py dashboard.wsgi:server`

4. **Deploy**: Railway will automatically deploy

The dashboard will be available at `https://your-app-name.up.railway.app`

---

### Option 3: Heroku

1. **Install Heroku CLI** from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login and create app**:
   ```bash
   heroku login
   heroku create mlb-win-probability-dashboard
   ```

3. **Set Python version**:
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

5. **Open**:
   ```bash
   heroku open
   ```

**Note**: Heroku free tier is no longer available. Paid plans start at $7/month.

---

### Option 4: AWS EC2 (Self-Hosted)

1. **Launch an EC2 instance**:
   - Choose Ubuntu 22.04 LTS
   - t2.micro or larger
   - Configure security group to allow HTTP (port 80) and HTTPS (port 443)

2. **SSH into the instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```

4. **Clone and setup**:
   ```bash
   git clone your-repo-url
   cd mlb-win-probability
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt gunicorn
   ```

5. **Run with Gunicorn**:
   ```bash
   gunicorn --config gunicorn_config.py dashboard.wsgi:server
   ```

6. **Setup Nginx** (reverse proxy):
   ```bash
   sudo nano /etc/nginx/sites-available/mlb-dashboard
   ```
   
   Add:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8050;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```
   
   Enable:
   ```bash
   sudo ln -s /etc/nginx/sites-available/mlb-dashboard /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

7. **Setup systemd service** (optional, for auto-start):
   ```bash
   sudo nano /etc/systemd/system/mlb-dashboard.service
   ```
   
   Add:
   ```ini
   [Unit]
   Description=MLB Win Probability Dashboard
   After=network.target
   
   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/mlb-win-probability
   Environment="PATH=/home/ubuntu/mlb-win-probability/venv/bin"
   ExecStart=/home/ubuntu/mlb-win-probability/venv/bin/gunicorn --config gunicorn_config.py dashboard.wsgi:server
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable:
   ```bash
   sudo systemctl enable mlb-dashboard
   sudo systemctl start mlb-dashboard
   ```

---

### Option 5: Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt gunicorn
   
   COPY . .
   
   EXPOSE 8050
   
   CMD ["gunicorn", "--config", "gunicorn_config.py", "dashboard.wsgi:server"]
   ```

2. **Build and run**:
   ```bash
   docker build -t mlb-dashboard .
   docker run -p 8050:8050 mlb-dashboard
   ```

3. **Deploy to Docker hosting**:
   - Docker Hub + Railway/Render
   - AWS ECS
   - Google Cloud Run
   - Azure Container Instances

---

## Important Notes

### Data Files
Ensure your deployment includes:
- `data/processed/games_processed.csv`
- `models/win_model_xgb.pkl` (or `win_model_lr.pkl`)
- `assets/logos/` directory with team logos
- `assets/mlb_logo.png`

You may need to:
1. Upload these files manually after deployment
2. Use a cloud storage service (S3, etc.) and modify code to load from there
3. Include them in your Git repository (if not too large)

### Environment Variables
For production, consider setting:
- `FLASK_ENV=production`
- `DEBUG=False`
- Any API keys or secrets

### Performance
- Free tiers may have limitations (memory, CPU, cold starts)
- Consider upgrading for better performance
- Use caching for data/model loading
- Optimize asset sizes

### Security
- Don't commit sensitive data
- Use environment variables for secrets
- Enable HTTPS (most platforms do this automatically)
- Set up proper authentication if needed

---

## Testing Locally Before Deployment

Test the production setup locally:

```bash
pip install gunicorn
gunicorn --config gunicorn_config.py dashboard.wsgi:server
```

Visit `http://localhost:8050` to verify it works.

---

## Troubleshooting

### Dashboard not loading
- Check logs on your hosting platform
- Verify all files are uploaded
- Check port configuration (should be 8050 or platform's assigned port)

### Model/data not found
- Ensure files are in the correct paths
- Check file permissions
- Verify paths in code match deployment structure

### Out of memory errors
- Upgrade to a larger instance/plan
- Reduce worker count in `gunicorn_config.py`
- Optimize data loading (use caching)

---

## Recommended Platform Comparison

| Platform | Ease | Cost | Performance | Best For |
|----------|------|------|-------------|----------|
| Render | ⭐⭐⭐⭐⭐ | Free tier available | Good | Quick deployment |
| Railway | ⭐⭐⭐⭐⭐ | Free tier available | Good | Easy setup |
| Heroku | ⭐⭐⭐⭐ | Paid only | Excellent | Established platform |
| AWS EC2 | ⭐⭐ | Pay-as-you-go | Excellent | Full control |
| Docker | ⭐⭐⭐ | Varies | Excellent | Containerized apps |

---

For questions or issues, check the platform's documentation or create an issue in your repository.

