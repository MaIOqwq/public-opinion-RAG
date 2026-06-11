# OpinionRAG - Game Opinion Intelligent Q&A System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Vue 3](https://img.shields.io/badge/Vue-3.4-4FC08D)](https://vuejs.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-orange)](https://www.trychroma.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-brightgreen)](https://deepseek.com/)

A Retrieval-Augmented Generation (RAG) system for intelligent Q&A about mobile game community opinions. Covers player discussions, comments, and videos from Bilibili and NGA platforms, supporting sentiment analysis, trend tracking, and controversy detection.

## Architecture

```mermaid
graph TD
    User([User]) -->|Query| Vue[Vue 3 + Element Plus Frontend]
    Vue -->|POST /api/rag/query| Flask[Flask API Backend]
    
    subgraph Backend Pipeline
        Flask -->|1. Structured Extraction| LLM[DeepSeek LLM]
        LLM -->|Filters| Flask
        Flask -->|2. Hybrid Retrieval| ChromaDB[(ChromaDB<br/>119K+ docs<br/>bge-small-zh-v1.5)]
        Flask -->|3. RRF Fusion| Rank[Reranking]
        Flask -->|4. Context Building| Context[Retrieval Context]
        Context -->|5. Prompt Assembly| LLM2[DeepSeek Chat]
        LLM2 -->|SSE Streaming| Flask
    end
    
    subgraph Data Pipeline
        MariaDB[(MariaDB<br/>Standardized Data)] -->|Vectorize| ChromaDB
        Crawler([Crawlers]) -->|Kafka/Spark| MariaDB
    end
    
    Flask -->|Streaming| Vue
    Vue -->|Display| User
```

## Features

- **Hybrid Retrieval**: Keyword pre-filtering + dense embedding re-ranking, overcoming ChromaDB's post-ANN WHERE limitations
- **LLM-Powered Structured Filtering**: Automatically extract game name, platform, time range, sentiment from natural language queries
- **Multi-Dimensional Scoring**: RRF fusion (dense + sparse), time decay, hot-score boost, deduplication, and diversity control
- **Gaming Slang Recognition**: Built-in mapping for community-specific terms (e.g., "节奏" = controversy, "长草" = content drought)
- **SSE Streaming**: Real-time token-by-token LLM response display
- **19 Tracked Mobile Games**: Genshin Impact, Honor of Kings, Honkai series, Zenless Zone Zero, Wuthering Waves, etc.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask + Gunicorn |
| Vector DB | ChromaDB (v0.4+) |
| Embedding Model | BAAI/bge-small-zh-v1.5 (384d) |
| LLM | DeepSeek Chat API |
| Sparse Retrieval | BM25 (rank_bm25) |
| Frontend | Vue 3 + Vite + Element Plus |
| Data Source | MariaDB (PyMySQL) |
| Deployment | Nginx + Systemd |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- ChromaDB data directory (build from scratch or download pre-built)

### Installation

```bash
# 1. Clone
git clone https://github.com/your-username/yulunrag.git
cd yulunrag

# 2. Backend setup
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example backend/.env
# Edit backend/.env to set DEEPSEEK_API_KEY

# 3. Frontend setup
cd frontend
npm install
npm run dev

# 4. Start backend (separate terminal)
cd backend
python app.py
```

## License

[MIT](LICENSE)
