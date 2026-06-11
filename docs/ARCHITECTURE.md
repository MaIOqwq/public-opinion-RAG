# Architecture Documentation

## Overview

OpinionRAG is a Retrieval-Augmented Generation system designed for game community opinion Q&A. It combines dense vector retrieval (ChromaDB + bge-small-zh-v1.5), sparse keyword retrieval (BM25), and a large language model (DeepSeek Chat) to answer questions about mobile game community sentiment, trends, and controversy events across Bilibili and NGA platforms.

## Retrieval Pipeline

The system employs a multi-stage retrieval pipeline to maximize relevance despite ChromaDB's architectural limitations:

```
User Question
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. Structured Filter Extraction (LLM)           │
│    ├── Game name detection                      │
│    ├── Platform detection (B站/NGA)             │
│    ├── Time range extraction                    │
│    ├── Sentiment polarity detection             │
│    └── Content type detection (post/comment/    │
│        video/reply)                             │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ 2. Query Building & Routing                     │
│                                                  │
│    IF game keyword detected:                     │
│      → Keyword pre-filter via ChromaDB get()     │
│      → Embedding re-ranking (cosine similarity)  │
│      (Works around ChromaDB's post-ANN WHERE)    │
│                                                  │
│    IF BM25 enabled (no specific keyword):        │
│      → Dense search (ChromaDB query)             │
│      → BM25 search (in-memory index)             │
│      → RRF fusion                                │
│      → Time decay × Hot boost multipliers        │
│      → Diversity cap per game                    │
│                                                  │
│    ELSE (dense-only fallback):                   │
│      → ChromaDB query with WHERE clause          │
│      → Post-filter by platform/type              │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ 3. Post-Search Filters                          │
│    ├── Metadata filter (keyword, platform, type)│
│    ├── Time range filter (string comparison)     │
│    └── Near-duplicate removal (3-gram Jaccard)   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ 4. Context Building                             │
│    ├── Aggregate stats (platform distribution,  │
│    │   date range, avg sentiment)               │
│    ├── Format entries with metadata              │
│    └── Trim to CONTEXT_MAX_LENGTH                │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ 5. LLM Generation                               │
│    ├── Prompt = template + context + question    │
│    ├── Stream via SSE (server-sent events)       │
│    └── Return sources metadata to frontend       │
└─────────────────────────────────────────────────┘
```

## Why Keyword Pre-Filter Instead of ChromaDB WHERE

ChromaDB v0.4.x implements WHERE as a **post-ANN filter**. When using `collection.query()` with a WHERE clause, ChromaDB first performs approximate nearest neighbor search, then filters results by metadata. This means:

1. If the ANN search doesn't return documents matching the keyword, the result is empty
2. With 119K+ documents across 19 games, keyword-matched documents may not rank high enough in ANN results
3. The `collection.get(where={"keyword": X})` approach uses **exact metadata filtering** first, then re-ranks by embedding similarity

The `search_by_keyword` method in `vector_db.py` implements this workaround:
```
collection.get(where={"keyword": keyword}, limit=fetch_k, include=["embeddings"])
→ Compute cosine similarity with query embedding
→ Sort by similarity score
→ Return top_k
```

## ChromaDB Data Schema

### Collection: `game_opinions`

**Document format:**
```
这是来自{platform}{board}板块的一条{type}，讨论{keyword}相关话题。
作者{author}发布于{publish_time}。
标题：{title}
正文：{content}
```

**Metadata fields:**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| keyword | string | "原神" | Game name tag |
| platform | string | "B站" | Source platform |
| type | string | "post" | Content type |
| sentiment_score | float | 0.75 | Sentiment score (0-1) |
| publish_time | string | "2026-05-15 10:30:00" | ISO datetime |
| hot_score | float | 2500.0 | Engagement metric |
| author | string | "用户A" | Author name |
| board_name | string | "综合" | Sub-forum name |
| (also: like_count, view_count, comment_count in some records) | | | |

## LLM Prompt Template

The system prompt includes:
1. **System role definition**: Expert game opinion analyst
2. **Gaming slang glossary**: Maps community terms to formal meanings (e.g., "节奏" = controversy)
3. **Retrieved context**: Aggregated search results with statistics
4. **User question**: The original query
5. **Answering rules**: Instructions for citation, honesty about data gaps, objectivity

## Hybrid Search: RRF (Reciprocal Rank Fusion)

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))

Where:
- rank_i(doc) = rank in search method i (dense or BM25)
- k = RRF constant (default: 60)
```

The RRF scores are then multiplied by:
- **Time decay**: `0.5^(days_since_publish / half_life)` — recent docs get higher score
- **Hot boost**: `1.0 + min(0.25, log10(hot_score) * 0.08)` — high-engagement docs get a boost

## BM25 Synonym Expansion

Before BM25 search, the query is expanded using a domain-specific synonym map:

| Term | Expanded Synonyms |
|------|-------------------|
| 负面 (negative) | 差评 吐槽 垃圾 坑 烂 恶心 翻车 暴死 |
| 节奏 (controversy) | 争议 炎上 骂战 喷 差评 负面 |
| 推荐 (recommend) | 好玩的 值得 入坑 安利 必玩 |
| 热门 (popular) | 火爆 最火 流行 热度高 讨论多 |

## Frontend Communication

- **Protocol**: Server-Sent Events (SSE)
- **Endpoint**: `POST /api/rag/query`
- **Event format**: `data: {"type": "chunk"|"done"|"error", "content": "...", "sources": [...]}`
- **Frontend**: Vue 3 uses `fetch()` + `ReadableStream` for incremental parsing
