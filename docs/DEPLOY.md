# Deployment Guide

## Production Deployment (Linux Server)

### Prerequisites

- Linux server (Ubuntu 22.04 recommended)
- Python 3.10+
- Nginx
- MariaDB (for data pipeline, optional for serving)
- ChromaDB data directory (119K+ documents)

### 1. Server Setup

```bash
# Install system dependencies
apt update && apt install -y python3 python3-venv python3-pip nginx

# Create application directory
mkdir -p /opt/rag-qa/{backend,frontend/dist,chroma_db,logs}
```

### 2. Backend Deployment

```bash
cd /opt/rag-qa

# Upload backend code (from local machine)
scp -r backend/app.py backend/config.py backend/vector_db.py backend/requirements.txt root@<SERVER_IP>:/opt/rag-qa/backend/
scp backend/.env root@<SERVER_IP>:/opt/rag-qa/backend/

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gunicorn
```

### 3. Frontend Build & Deploy

```bash
# Build locally (from project root)
cd frontend
npm install
npm run build

# Upload dist to server
scp -r frontend/dist/* root@<SERVER_IP>:/opt/rag-qa/frontend/dist/
```

### 4. ChromaDB Data

```bash
# Upload ChromaDB persistent data
scp -r chroma_db/* root@<SERVER_IP>:/opt/rag-qa/chroma_db/

# Or build from MariaDB data (if you have the data pipeline)
python scripts/vectorize_data.py \
    --host <DB_HOST> --port 3306 \
    --user <DB_USER> --password <DB_PASSWORD> \
    --database <DB_NAME> \
    --chroma-path /opt/rag-qa/chroma_db
```

### 5. Gunicorn Configuration

The service uses Gunicorn with the following settings:

- **Worker**: 1 (single worker, ChromaDB is not thread-safe for writes)
- **Bind**: 127.0.0.1:8002 (reverse-proxied by Nginx)
- **Timeout**: 300s (for long LLM responses)
- **Graceful timeout**: 30s
- **Preload**: Not enabled (use `post_fork` hook for ChromaDB reconnection)

The `post_fork` function in `app.py` reinitializes the ChromaDB client after each worker fork to prevent SQLite database locked errors.

### 6. Systemd Service

Create `/etc/systemd/system/rag-qa.service` using the template from `deploy/rag-qa.service`:

```bash
cp deploy/rag-qa.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable rag-qa
systemctl start rag-qa
systemctl status rag-qa
```

### 7. Nginx Reverse Proxy

Configure Nginx using the template from `deploy/rag-qa.nginx.conf`:

```nginx
server {
    listen 80;
    server_name <SERVER_IP>;

    # Frontend static files
    location /qa {
        alias /opt/rag-qa/frontend/dist;
        index index.html;
        try_files $uri $uri/ /qa/index.html;
    }

    # Backend API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;       # Required for SSE
        proxy_cache off;
        proxy_read_timeout 60s;
    }
}
```

```bash
cp deploy/rag-qa.nginx.conf /etc/nginx/sites-available/rag-qa
ln -sf /etc/nginx/sites-available/rag-qa /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## Environment Variables

See `.env.example` for all configurable variables. Key variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | - | DeepSeek API key |
| `CHROMA_PATH` | No | `./chroma_db` | ChromaDB persist directory |
| `FLASK_ENV` | No | `development` | Set to `production` in production |
| `BM25_ENABLED` | No | `true` | Enable hybrid BM25 + dense search |
| `TOP_K` | No | `15` | Number of documents in context |

## Monitoring

### Health Check

```bash
curl http://localhost:8002/api/health
```

Response:
```json
{
    "status": "healthy",
    "collection_info": {
        "document_count": 119000,
        "bm25_ready": true
    },
    "llm_provider": "deepseek",
    "bm25_enabled": true
}
```

### Logs

```bash
tail -f /opt/rag-qa/logs/app.log
```

### Restart

```bash
systemctl restart rag-qa
```

## Offline Mode

If the server has no internet access, set `HF_ENDPOINT` to a mirror:

```ini
# In systemd service or .env
HF_ENDPOINT=https://hf-mirror.com
```

This is needed because SentenceTransformer downloads the model from Hugging Face on first run.

## Troubleshooting

### ChromaDB "database is locked"
- Ensure only one gunicorn worker is used (`-w 1`)
- The `post_fork` hook reconnects ChromaDB after fork

### BM25 "not ready yet" → falling back to dense-only
- BM25 index builds asynchronously on first query
- Run `python scripts/build_bm25_cache.py` offline to pre-build

### SSE not working
- Ensure `proxy_buffering off` in nginx config
- Check `X-Accel-Buffering: no` header is set
