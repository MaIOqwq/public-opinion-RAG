# Changelog

## [1.0.0] - 2026-06-11

### Added
- Initial open-source release of OpinionRAG
- Hybrid retrieval pipeline (dense + BM25 + RRF fusion)
- LLM-powered structured filter extraction from natural language queries
- Keyword pre-filter + embedding re-ranking (workaround for ChromaDB post-ANN WHERE)
- Time decay and hot-score boosting for result ranking
- Near-duplicate removal via 3-gram Jaccard similarity
- Diversity cap to prevent single-game domination
- BM25 synonym expansion for gaming domain terms
- SSE streaming for real-time LLM responses
- Vue 3 + Element Plus chat interface
- Gunicorn + Nginx + Systemd deployment configuration
- MariaDB-to-ChromaDB vectorization pipeline
- ChromaDB export/import utilities
- Complete documentation (Chinese + English)
