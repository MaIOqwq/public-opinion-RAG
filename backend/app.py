import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import pymysql

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

# Alias map: slang/abbreviation → official keyword
GAME_ALIASES = {
    "吃鸡": "和平精英",
    "农药": "王者荣耀",
    "lol手游": "英雄联盟手游",
    "联盟手游": "英雄联盟手游",
    "三角洲": "三角洲行动",
    "方舟": "明日方舟",
    "终末地": "明日方舟终末地",
    "星铁": "崩坏星穹铁道",
    "崩铁": "崩坏星穹铁道",
    "三崩子": "崩坏三",
    "第五": "第五人格",
    "暖暖": "无限暖暖",
    "逆水寒": "逆水寒手游",
    "燕云": "燕云十六声",
    "绝区": "绝区零",
    "zzz": "绝区零",
    "火影": "火影忍者手游",
    "永劫": "永劫无间手游",
    "金铲铲": "金铲铲之战",
    "铲铲": "金铲铲之战",
}


def _extract_keyword_local(question: str) -> Optional[str]:
    """Extract game keyword via local string matching.
    Two-pass: exact match first, then alias/substring match.
    Returns None if no match."""
    # Pass 1: exact full-name match (most reliable, handles compound names)
    for kw in GAME_KEYWORDS:
        if kw in question:
            logger.info("Keyword (exact match): %s", kw)
            return kw

    # Pass 2: alias match
    for alias, official in GAME_ALIASES.items():
        if alias in question:
            logger.info("Keyword (alias: %s -> %s)", alias, official)
            return official

    return None


def _extract_keyword_llm(question: str) -> Optional[str]:
    """Fallback: use LLM to extract game name.
    Only called when local matching fails (e.g., new aliases, typos)."""
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": (
                    "You are a game name extractor. Extract the video game name"
                    " mentioned in the user's question.\n"
                    f"Known game list: {', '.join(GAME_KEYWORDS)}\n"
                    'Output ONLY: {"keyword": "game_name"} or {"keyword": null}\n'
                    "No other output."
                )
            }, {
                "role": "user",
                "content": question
            }],
            temperature=0,
            max_tokens=30,
            timeout=5,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        kw = result.get("keyword")
        if kw and isinstance(kw, str) and kw.strip():
            kw = kw.strip()
            for gk in GAME_KEYWORDS:
                if kw == gk:
                    logger.info("Keyword (LLM): %s", kw)
                    return kw
        return None
    except Exception as e:
        logger.warning("Keyword LLM extraction failed: %s", str(e))
        return None


def _extract_platform_local(question: str) -> Optional[str]:
    if "B站" in question or "b站" in question or "bilibili" in question.lower():
        return "B站"
    if "NGA" in question or "nga" in question.lower():
        return "NGA"
    return None


def _extract_time_local(question: str) -> tuple:
    """Returns (time_after, time_before) or (None, None)."""
    import re
    now = datetime.now()
    if re.search(r'昨天', question):
        d = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        return d, d
    elif re.search(r'今天|今日', question):
        d = now.strftime('%Y-%m-%d')
        return d, None
    elif re.search(r'最近一周|近7天|这周|本周|最近七天', question):
        return (now - timedelta(days=7)).strftime('%Y-%m-%d'), None
    elif re.search(r'最近三天|近3天', question):
        return (now - timedelta(days=3)).strftime('%Y-%m-%d'), None
    elif re.search(r'最近一个月|近30天|这一个月', question):
        return (now - timedelta(days=30)).strftime('%Y-%m-%d'), None
    return None, None


def _extract_sentiment_local(question: str) -> tuple:
    """Returns (sentiment_min, sentiment_max) or (None, None)."""
    import re
    neg = re.search(r'节奏|争议|喷|冲|炎上|骂|负面|差评|吐槽|翻车|暴死|烂', question)
    pos = re.search(r'吹爆|好评|正面|良心|推荐|安利|好玩|必玩', question)
    if neg:
        return None, 0.4
    if pos:
        return 0.6, None
    return None, None


# ── Pipeline: step 1 – Structured filter extraction ──

def _build_search_query(question: str, filters: dict) -> str:
    kw = (filters or {}).get("keyword")
    if kw:
        return f"{kw} {kw} {kw} {question}"
    return question


def _extract_filters(question: str) -> dict:
    """Hybrid extraction: local regex for keyword/platform/time/sentiment (fast & reliable),
    LLM enrichment for search_terms and complex queries (optional)."""
    if not Config.STRUCTURED_FILTER_ENABLED:
        return {}

    # Step 1: Local extraction (fast, deterministic)
    keyword = _extract_keyword_local(question)
    platform = _extract_platform_local(question)
    time_after, time_before = _extract_time_local(question)
    sentiment_min, sentiment_max = _extract_sentiment_local(question)

    result = {
        "keyword": keyword,
        "platform": platform,
        "time_after": time_after,
        "time_before": time_before,
        "sentiment_max": sentiment_max,
        "sentiment_min": sentiment_min,
        "type": None,
        "search_terms": None,
    }

    # Step 2: LLM enrichment for search_terms and type only (lightweight)
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": (
                    "从用户问题提取补充搜索条件。只输出JSON。\n"
                    "字段:\n"
                    "- search_terms: 扩展用户意图的搜索词(空格分隔)。未提及填null\n"
                    "规则: \"排行/最火/热度最高\" -> search_terms=\"热度 热门\"\n"
                    "只输出JSON，不要解释"
                )
            }, {
                "role": "user",
                "content": question
            }],
            temperature=0.1,
            max_tokens=80,
            timeout=5,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        enriched = json.loads(raw)
        for field in ["search_terms"]:
            val = enriched.get(field)
            if val is not None:
                result[field] = val
    except Exception as e:
        logger.warning("Enrichment failed: %s", str(e))

    # Step 3: If keyword still null and STRICT mode, try LLM as last resort
    if not keyword:
        llm_kw = _extract_keyword_llm(question)
        if llm_kw:
            result["keyword"] = llm_kw

    logger.info("Filters: %s", json.dumps(result, ensure_ascii=False))
    return result


# ── ChromaDB WHERE clause builder ──

def _build_where(filters: dict) -> Optional[Dict[str, Any]]:
    """Convert extracted filters to ChromaDB WHERE clause (sentiment only).
    keyword/platform/type done in _filter_by_meta for reliability."""
    conditions = []
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


# ── Text-to-SQL for keyword-less queries ──

MYSQL_SCHEMA = """
表名: standardized_data
字段:
  keyword (VARCHAR): 游戏名称 (注意: "手机游戏" 是泛用标签不是具体游戏，查询时过滤掉)
  platform (TINYINT): 0=B站 1=NGA
  type (VARCHAR): post/video/comment/reply
  author (VARCHAR): 作者名
  title_clean (TEXT): 标题
  content_clean (TEXT): 正文内容
  publish_time (DATETIME): 发布时间
  hot_score (DOUBLE): 热度分数
  sentiment_score (DOUBLE): 情感分数 0~1, >0.5偏正面 <0.5偏负面
  view_count (INT): 播放/阅读数
  like_count (INT): 点赞数
  comment_count (INT): 评论数
  board_name (VARCHAR): 板块名
"""

SQL_PROMPT = """你是一个MariaDB SQL专家。根据用户问题和下表结构，生成一条SQL查询。

{schema}

用户问题: {question}

要求:
1. 只输出SQL语句，不要任何解释，不要markdown代码块
2. 只生成SELECT语句，禁止INSERT/UPDATE/DELETE/DROP
3. 涉及多个游戏比较/排名时用 GROUP BY keyword
4. 排序用 ORDER BY，限制用 LIMIT
5. 时间筛选用 publish_time，平台: platform=0(B站) platform=1(NGA)
6. 情感筛选用 sentiment_score
7. LIMIT 不超过 50"""


def _text_to_sql(question: str) -> Optional[str]:
    """Ask LLM to generate SQL from natural language question."""
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": SQL_PROMPT.format(schema=MYSQL_SCHEMA, question=question)
            }],
            temperature=0,
            max_tokens=400,
            timeout=10,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
        raw = raw.strip()

        if not raw.upper().startswith("SELECT"):
            logger.warning("Text-to-SQL returned non-SELECT: %s", raw[:100])
            return None
        for kw in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]:
            if kw in raw.upper():
                logger.warning("Text-to-SQL returned dangerous SQL: %s", raw[:100])
                return None

        logger.info("Text-to-SQL: %s", raw)
        return raw
    except Exception as e:
        logger.warning("Text-to-SQL LLM call failed: %s", str(e))
        return None


def _execute_sql(sql: str) -> tuple:
    """Execute SQL against MariaDB. Returns (rows: list of dict, error: str or None)."""
    conn = None
    try:
        conn = pymysql.connect(
            host=Config.MYSQL_HOST, port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER, password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
            charset='utf8mb4', connect_timeout=3, read_timeout=10,
        )
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:
        logger.error("SQL execution failed: %s", str(e))
        return [], str(e)
    finally:
        if conn:
            conn.close()


def _build_sql_result_context(rows: List[dict], question: str) -> str:
    """Format SQL results as readable context for LLM."""
    if not rows:
        return "【数据库查询结果】未找到匹配数据"

    lines = [f"【数据库查询结果】共{len(rows)}条："]
    for i, r in enumerate(rows, 1):
        parts = []
        for k, v in r.items():
            if v is None:
                continue
            if isinstance(v, float):
                v = round(v, 3)
            elif hasattr(v, 'strftime'):
                v = v.strftime('%Y-%m-%d %H:%M')
            elif isinstance(v, str) and len(str(v)) > 80:
                v = str(v)[:80] + "..."
            parts.append(f"{k}={v}")
        lines.append(f"[{i}] " + " | ".join(parts))

    return "\n".join(lines)


def _filter_by_time(results, time_after=None, time_before=None):
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

    filters = _extract_filters(question)
    keyword = filters.get("keyword") if filters else None

    # Path B: No keyword → Text-to-SQL
    if not keyword:
        sql_rows = []
        sql = _text_to_sql(question)
        if sql:
            rows, sql_error = _execute_sql(sql)
            if rows and not sql_error:
                sql_rows = rows
                logger.info("Text-to-SQL returned %d rows", len(rows))

        if sql_rows:
            sql_context = _build_sql_result_context(sql_rows, question)
            prompt = Config.PROMPT_TEMPLATE.format(context=sql_context, question=question)
            sources = [f"SQL: {sql}"]

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
                    logger.info("Text-to-SQL done in %.2fs | answer=%d chars", elapsed, len(answer))
                    yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'answer': answer})}\n\n"
                except Exception as e:
                    logger.error("Text-to-SQL LLM error: %s", str(e))
                    yield f"data: {json.dumps({'type': 'error', 'content': '查询失败请稍后重试'})}\n\n"

            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )
        # SQL failed → fall through to dense search below

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

    search_results = _filter_by_meta(search_results, filters)

    if time_after or time_before:
        search_results = _filter_by_time(search_results, time_after, time_before)

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
    })


@app.route('/api/rag/debug_search', methods=['POST'])
def debug_search():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    filters = _extract_filters(question)
    keyword = filters.get("keyword") if filters else None

    # Path B: No keyword → Text-to-SQL
    if not keyword:
        sql_rows = []
        sql = _text_to_sql(question)
        if sql:
            rows, sql_error = _execute_sql(sql)
            if rows and not sql_error:
                sql_rows = rows

        if sql_rows:
            return jsonify({
                "success": True,
                "mode": "text_to_sql",
                "sql": sql,
                "total": len(sql_rows),
                "filters": filters,
                "results": sql_rows,
            })
        # SQL failed → fall through to dense search below

    where = _build_where(filters)
    search_terms = filters.get("search_terms") if filters else None

    time_after = filters.get("time_after") if filters else None
    time_before = filters.get("time_before") if filters else None
    has_other = bool((filters or {}).get("platform") or (filters or {}).get("type") or time_after or time_before)
    fetch_k = Config.HYBRID_TOP_K
    if has_other:
        fetch_k = max(fetch_k, 200)
    if time_after or time_before:
        fetch_k = max(fetch_k, 10000)

    search_query = _build_search_query(question, filters)
    if keyword:
        results = vector_db.search_by_keyword(search_query, keyword, top_k=fetch_k, fetch_k=fetch_k)
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
