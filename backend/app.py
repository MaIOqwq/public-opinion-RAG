import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

from config import Config
from vector_db import VectorDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

vector_db = VectorDBManager(
    chroma_path=Config.CHROMA_PATH,
    collection_name=Config.COLLECTION_NAME,
    embedding_model=Config.EMBEDDING_MODEL,
    bm25_enabled=Config.BM25_ENABLED,
)

llm_client = OpenAI(
    api_key=Config.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)


# ── Gunicorn post-fork hook ──

def post_fork(server, worker):
    """Reinitialize ChromaDB client after fork (--preload safe)."""
    vector_db.reconnect()


# ── Known game keywords (from DB) ──
GAME_KEYWORDS = [
    "三角洲行动", "光遇", "原神", "和平精英", "天涯明月刀",
    "崩坏三", "崩坏星穹铁道", "手机游戏", "无限暖暖", "明日方舟",
    "明日方舟终末地", "永劫无间手游", "火影忍者手游", "燕云十六声",
    "王者荣耀", "第五人格", "绝区零", "英雄联盟手游", "逆水寒手游",
    "金铲铲之战", "阴阳师", "鸣潮",
]


# ── Pipeline: step 1 – Structured filter extraction ──

def _build_search_query(question: str, filters: dict) -> str:
    """Inject game keyword into query to boost dense embedding signal.
    bge-small-zh-v1.5 prioritizes semantic similarity over keyword matching.
    Prepending the keyword 3x makes the embedding vector more keyword-aware."""
    kw = (filters or {}).get("keyword")
    if kw:
        return f"{kw} {kw} {kw} {question}"
    return question


def _extract_filters(question: str) -> dict:
    """LLM extracts structured ChromaDB WHERE conditions from user question.
    Returns empty dict on failure (caller falls back to unfiltered search)."""
    if not Config.STRUCTURED_FILTER_ENABLED:
        return {}
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": f"""你是搜索条件提取器。从用户问题提取结构化过滤条件，输出JSON。

今天是{datetime.now().strftime("%Y年%m月%d日")}。所有时间相关的计算必须以今天为基准。

可用字段:
- keyword: 游戏名，必须是以下之一: {", ".join(GAME_KEYWORDS)}
  如果用户没提具体游戏名，填null
- platform: "B站" 或 "NGA"。未提及填null
- time_after: 开始日期 "YYYY-MM-DD"。未提及填null
- time_before: 结束日期 "YYYY-MM-DD"。未提及填null
- sentiment_max: 0.0~1.0，查负面内容时设<=0.4。未提及填null
- sentiment_min: 0.0~1.0，查正面内容时设>=0.6。未提及填null
- type: "post"/"comment"/"video"/"reply"。未提及填null
- search_terms: 搜索关键词补充(空格分隔)，扩展用户意图的同义词

规则:
1. 只有明确的时间限定才设time_after: "最近一周/近7天/这周" -> 7天前; "昨天" -> time_after和time_before都设昨天; "今天" -> time_after设今天
   注意: "最近有什么"/"最近讨论"/"最近热门"这类泛指"当前"但不限时间的，不要设time_after
3. "今天" -> time_after设为今天
4. "节奏/争议/喷/冲/负面/骂/炎上" -> sentiment_max=0.4
5. "吹爆/好评/正面/良心" -> sentiment_min=0.6
6. "视频" -> type="video"
7. "最火/排行/热度最高" -> 不设keyword，search_terms填"热度 热门 排行"
8. 未提及的字段填null
9. 只输出JSON，不要任何解释"""
            }, {
                "role": "user",
                "content": question
            }],
            temperature=0.1,
            max_tokens=150,
            timeout=5,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        # Fallback: if LLM missed the game name, simple string match
        if not result.get("keyword"):
            for kw in GAME_KEYWORDS:
                if kw in question:
                    result["keyword"] = kw
                    break
        logger.info("Filters: %s", json.dumps(result, ensure_ascii=False))
        return result
    except Exception as e:
        logger.warning("Filter extraction failed (fallback to unfiltered): %s", str(e))
        return {}


def _build_where(filters: dict) -> Optional[Dict[str, Any]]:
    """Convert extracted filters to ChromaDB WHERE clause.
    keyword/platform/type moved to post-filter for reliability (ChromaDB 1.5.9 buggy)."""
    conditions = []
    # Sentiment only — keyword/platform/type done in _filter_by_meta
    sent = {}
    if filters.get("sentiment_max") is not None:
        sent["$lte"] = float(filters["sentiment_max"])
    if filters.get("sentiment_min") is not None:
        sent["$gte"] = float(filters["sentiment_min"])
    if sent:
        conditions.append({"sentiment_score": sent})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Pipeline: step 2-4 – Context building ──

def _filter_by_time(results, time_after=None, time_before=None):
    """Post-search time filter. ChromaDB 1.5.9 doesn't support string $gte/$lte."""
    if not time_after and not time_before:
        return results
    filtered = []
    for r in results:
        pub = (r.get("metadata", {}).get("publish_time", "") or "")[:10]
        if not pub:
            filtered.append(r)
            continue
        if time_after and pub < time_after:
            continue
        if time_before and pub > time_before:
            continue
        filtered.append(r)
    return filtered


def _filter_by_meta(results, filters):
    """Post-search meta filter. keyword/platform/type — more reliable than ChromaDB WHERE."""
    keep = []
    for r in results:
        meta = r.get("metadata", {})
        keyword = str(filters.get("keyword") or "")
        platform = str(filters.get("platform") or "")
        dtype = str(filters.get("type") or "")
        if keyword and str(meta.get("keyword") or "") != keyword:
            continue
        if platform and str(meta.get("platform") or "") != platform:
            continue
        if dtype and str(meta.get("type") or "") != dtype:
            continue
        keep.append(r)
    return keep


def _dedup_results(results: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
    if len(results) <= 1:
        return results

    def _ngrams(text: str, n: int = 3) -> set:
        t = text[:300]
        return {t[i:i+n] for i in range(len(t) - n + 1)}

    kept = []
    for r in results:
        doc = r.get("document", "")
        r_grams = _ngrams(doc)
        if not r_grams:
            kept.append(r)
            continue
        dup = False
        for k in kept:
            k_grams = _ngrams(k.get("document", ""))
            if not k_grams:
                continue
            if len(r_grams & k_grams) / len(r_grams | k_grams) > threshold:
                dup = True
                break
        if not dup:
            kept.append(r)
    logger.info("Dedup: %d -> %d", len(results), len(kept))
    return kept


def build_context(results, max_length=2000):
    stats_parts = [f'共{len(results)}条']
    platforms = {}
    dates = []
    sentiments = []
    for r in results:
        m = r.get("metadata", {})
        plat = m.get("platform", "?")
        platforms[plat] = platforms.get(plat, 0) + 1
        pub = m.get("publish_time", "")[:10]
        if pub:
            dates.append(pub)
        sent = m.get("sentiment_score")
        if sent is not None:
            sentiments.append(float(sent))
    stats_parts.append(" | ".join(f"{k}:{v}" for k, v in platforms.items()))
    if dates:
        dates.sort()
        stats_parts.append(f"{dates[0]}~{dates[-1]}")
    if sentiments:
        avg = sum(sentiments) / len(sentiments)
        label = "偏负面" if avg < 0.45 else ("偏正面" if avg > 0.55 else "中性")
        stats_parts.append(f"均情感:{avg:.2f}({label})")
    header = "【检索统计】" + " | ".join(stats_parts)

    header_len = len(header) + 3
    remaining = max_length - header_len
    entries = []
    current = 0
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        plat = meta.get("platform", "?")
        board = meta.get("board_name", "")
        kw = meta.get("keyword", "")
        ptime = meta.get("publish_time", "")[:10]
        hot = round(meta.get("hot_score", 0), 3)
        sent = round(meta.get("sentiment_score", 0.5), 2)
        likes = meta.get("like_count", 0) or 0
        views = meta.get("view_count", 0) or 0
        comments = meta.get("comment_count", 0) or 0

        meta_line = f"[{i}] {plat}"
        if board:
            meta_line += f" | {board}"
        if kw:
            meta_line += f" | {kw}"
        if ptime:
            meta_line += f" | {ptime}"
        meta_line += f" | 情感{sent}"
        if hot:
            meta_line += f" | 热度{hot}"
        if likes:
            meta_line += f" | 赞{likes}"
        if views:
            meta_line += f" | 播放{views}"
        if comments:
            meta_line += f" | 评论{comments}"

        entry = meta_line + "\n" + r["document"]
        if current + len(entry) <= remaining:
            entries.append(entry)
            current += len(entry)
        else:
            break

    return header + "\n\n" + "\n\n".join(entries)


def format_source(meta: dict) -> str:
    parts = [
        meta.get("platform", ""),
        meta.get("board_name", "") + "板块" if meta.get("board_name") else "",
        meta.get("type", ""),
        "关键词:" + meta.get("keyword", ""),
        "作者:" + meta.get("author", ""),
        "时间:" + meta.get("publish_time", ""),
        "热度:" + str(round(meta.get("hot_score", 0), 3)),
        "情感:" + str(round(meta.get("sentiment_score", 0), 2)),
        "赞:" + str(meta.get("like_count", 0) or 0),
    ]
    return " | ".join(p for p in parts if p)


# ── Routes ──

@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    start_time = time.time()
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    logger.info("Query: %s", question)

    # 1. Structured filter extraction
    filters = _extract_filters(question)
    where = _build_where(filters)
    search_terms = filters.get("search_terms") if filters else None

    # 2. Search
    # Keyword uses collection.get() pre-filter + embedding re-rank (works around
    # ChromaDB WHERE being post-ANN filter, which yields too few keyword matches).
    # Other filters (platform/type) are post-filtered by _filter_by_meta.
    time_after = filters.get("time_after") if filters else None
    time_before = filters.get("time_before") if filters else None
    keyword = filters.get("keyword") if filters else None
    has_other = bool((filters or {}).get("platform") or (filters or {}).get("type") or time_after or time_before)
    fetch_k = Config.HYBRID_TOP_K
    if has_other:
        fetch_k = max(fetch_k, 200)
    if time_after or time_before:
        fetch_k = max(fetch_k, 10000)  # Time filter post-retrieval: wide pool needed

    search_query = _build_search_query(question, filters)
    if keyword:
        search_results = vector_db.search_by_keyword(search_query, keyword, top_k=fetch_k, fetch_k=fetch_k)
    elif Config.BM25_ENABLED:
        search_results = vector_db.hybrid_search(
            query=search_query,
            top_k=fetch_k,
            rrf_k=Config.RRF_K,
            where=where,
            time_decay_enabled=Config.TIME_DECAY_ENABLED,
            hot_boost_enabled=Config.HOT_BOOST_ENABLED,
            diversity_enabled=Config.DIVERSITY_ENABLED,
            diversity_max_per_game=Config.DIVERSITY_MAX_PER_GAME,
            bm25_query_override=search_terms,
            query_expansion_enabled=Config.QUERY_EXPANSION_ENABLED,
        )
    else:
        search_results = vector_db.search_dense(search_query, fetch_k, where)

    # 3. Post-search meta filter (keyword/platform/type — more reliable than ChromaDB WHERE)
    search_results = _filter_by_meta(search_results, filters)

    # 4. Post-search time filter (ChromaDB WHERE doesn't support string dates)
    if time_after or time_before:
        search_results = _filter_by_time(search_results, time_after, time_before)

    # 5. Dedup + trim
    if Config.DEDUP_ENABLED:
        search_results = _dedup_results(search_results)
    search_results = search_results[:Config.TOP_K]

    if not search_results:
        return jsonify({
            "success": True,
            "answer": "根据现有知识库无法回答该问题",
            "sources": [],
            "filters": filters,
        })

    # 4. Build context + generate
    context = build_context(search_results, Config.CONTEXT_MAX_LENGTH)
    prompt = Config.PROMPT_TEMPLATE.format(context=context, question=question)
    sources = [format_source(r["metadata"]) for r in search_results]

    def generate():
        full_answer = []
        try:
            stream = llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                stream=True,
                timeout=30,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer.append(delta)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': delta})}\n\n"

            elapsed = time.time() - start_time
            answer = "".join(full_answer)
            logger.info("Done in %.2fs | answer=%d chars | filters=%s",
                       elapsed, len(answer), json.dumps(filters, ensure_ascii=False))

            yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'answer': answer})}\n\n"

        except Exception as e:
            logger.error("LLM stream error: %s", str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': '查询失败请稍后重试'})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    info = vector_db.get_collection_info()
    return jsonify({
        "status": "healthy",
        "collection_info": info,
        "llm_provider": Config.LLM_PROVIDER,
        "bm25_enabled": Config.BM25_ENABLED,
        "structured_filter": Config.STRUCTURED_FILTER_ENABLED,
        "query_expansion": Config.QUERY_EXPANSION_ENABLED,
        "time_decay": Config.TIME_DECAY_ENABLED,
        "hot_boost": Config.HOT_BOOST_ENABLED,
        "dedup": Config.DEDUP_ENABLED,
        "diversity": Config.DIVERSITY_ENABLED,
    })


@app.route('/api/rag/debug_search', methods=['POST'])
def debug_search():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    filters = _extract_filters(question)
    where = _build_where(filters)
    search_terms = filters.get("search_terms") if filters else None

    time_after = filters.get("time_after") if filters else None
    time_before = filters.get("time_before") if filters else None
    keyword = filters.get("keyword") if filters else None
    has_other = bool((filters or {}).get("platform") or (filters or {}).get("type") or time_after or time_before)
    fetch_k = Config.HYBRID_TOP_K
    if has_other:
        fetch_k = max(fetch_k, 200)
    if time_after or time_before:
        fetch_k = max(fetch_k, 10000)

    search_query = _build_search_query(question, filters)
    if keyword:
        results = vector_db.search_by_keyword(search_query, keyword, top_k=fetch_k)
    elif Config.BM25_ENABLED:
        results = vector_db.hybrid_search(
            query=search_query,
            top_k=fetch_k,
            rrf_k=Config.RRF_K,
            where=where,
            time_decay_enabled=Config.TIME_DECAY_ENABLED,
            hot_boost_enabled=Config.HOT_BOOST_ENABLED,
            diversity_enabled=Config.DIVERSITY_ENABLED,
            diversity_max_per_game=Config.DIVERSITY_MAX_PER_GAME,
            bm25_query_override=search_terms,
            query_expansion_enabled=Config.QUERY_EXPANSION_ENABLED,
        )
    else:
        results = vector_db.search_dense(search_query, fetch_k, where)

    results = _filter_by_meta(results, filters)

    if time_after or time_before:
        results = _filter_by_time(results, time_after, time_before)

    return jsonify({
        "success": True,
        "total": len(results),
        "filters": filters,
        "where": where,
        "results": [{
            "id": r.get("id", ""),
            "document": r["document"][:500],
            "metadata": r["metadata"],
            "score": r["score"],
        } for r in results[:10]]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=Config.DEBUG, use_reloader=False)
