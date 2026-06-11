#!/bin/bash
# OpinionRAG deployment script
# Usage: ./deploy.sh [SSH_KEY_PATH] [SERVER] [LOCAL_BASE] [REMOTE_BASE]

SSH_KEY="${1:-<SSH_KEY_PATH>}"
SERVER="${2:-root@<SERVER_IP>}"
LOCAL_BASE="${3:-<PROJECT_ROOT>}"
REMOTE_BASE="${4:-/opt/rag-qa}"

echo "=== 1/6 Creating remote directories ==="
ssh -i "$SSH_KEY" "$SERVER" "mkdir -p $REMOTE_BASE/backend $REMOTE_BASE/frontend/dist $REMOTE_BASE/chroma_db $REMOTE_BASE/logs"

echo "=== 2/6 Uploading backend code ==="
scp -i "$SSH_KEY" "$LOCAL_BASE/backend/app.py" "${SERVER}:$REMOTE_BASE/backend/"
scp -i "$SSH_KEY" "$LOCAL_BASE/backend/config.py" "${SERVER}:$REMOTE_BASE/backend/"
scp -i "$SSH_KEY" "$LOCAL_BASE/backend/vector_db.py" "${SERVER}:$REMOTE_BASE/backend/"
scp -i "$SSH_KEY" "$LOCAL_BASE/backend/requirements.txt" "${SERVER}:$REMOTE_BASE/backend/"
scp -i "$SSH_KEY" "$LOCAL_BASE/backend/.env" "${SERVER}:$REMOTE_BASE/backend/"

echo "=== 3/6 Uploading ChromaDB data ==="
scp -i "$SSH_KEY" -r "$LOCAL_BASE/chroma_db/*" "${SERVER}:$REMOTE_BASE/chroma_db/"

echo "=== 4/6 Uploading frontend dist ==="
scp -i "$SSH_KEY" -r "$LOCAL_BASE/frontend/dist/*" "${SERVER}:$REMOTE_BASE/frontend/dist/"

echo "=== 5/6 Installing Python dependencies + configuring service ==="
ssh -i "$SSH_KEY" "$SERVER" "cd $REMOTE_BASE && python3 -m venv venv && ./venv/bin/pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=== 6/6 Configuring systemd service ==="
scp -i "$SSH_KEY" "$LOCAL_BASE/deploy/rag-qa.service" "${SERVER}:/etc/systemd/system/"
ssh -i "$SSH_KEY" "$SERVER" "systemctl daemon-reload && systemctl enable rag-qa && systemctl start rag-qa"

echo "=== Deployment complete ==="
