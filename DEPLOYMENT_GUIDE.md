# VISION-AI Deployment Guide

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run individual container
docker build -t vision-ai .
docker run -p 3000:3000 -p 8000:8000 vision-ai
```

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start main application
python3 start_app.py

# Start PVLA system (in another terminal)
cd QEP-VLA-App
python3 simple_web_server.py
```

## Access Points

- **Main VisionA App**: http://localhost:3000
- **PVLA Navigation System**: http://localhost:8000
- **ARIA AI Demo**: http://localhost:8000/aria-demo
- **API Documentation**: http://localhost:8000/docs

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill processes using ports 3000 or 8000
   lsof -ti:3000 | xargs kill -9
   lsof -ti:8000 | xargs kill -9
   ```

2. **Docker build fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   docker build --no-cache -t vision-ai .
   ```

3. **Missing dependencies**
   ```bash
   # Reinstall requirements
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Permission issues**
   ```bash
   # Fix file permissions
   chmod +x start_app.py
   chmod +x start_app.sh
   ```

### Health Checks

- Main app: `curl http://localhost:3000/`
- PVLA system: `curl http://localhost:8000/health`
- Docker health: `docker ps` (check STATUS column)

## Production Deployment

### Environment Variables
```bash
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1
export ENVIRONMENT=production
```

### Docker Production
```bash
# Build production image
docker build -t vision-ai:latest .

# Run with production settings
docker run -d \
  --name vision-ai \
  -p 3000:3000 \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  vision-ai:latest
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f QEP-VLA-App/deploy/k8s/
```

## Monitoring

- **Logs**: Check `./logs/` directory
- **Health**: Use health check endpoints
- **Metrics**: Prometheus metrics available at `/metrics`

## Support

For deployment issues, check:
1. This deployment guide
2. Application logs
3. Docker/Kubernetes logs
4. Health check endpoints
