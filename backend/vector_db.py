import logging
import math
import os
import pickle
import jieba
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import threading

jieba.setLogLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Domain synonym map ──
SYNONYM_MAP = {
    "负面": "差评 吐槽 垃圾 坑 烂 恶心 翻车 暴死",
    "风评": "口碑 评价 看法 舆论",
    "竞品": "同类游戏 同类 类似 竞争 对比",
    "热门": "火爆 最火 流行 热度高 讨论多",
    "更新": "版本 新内容 新角色 新活动 上线",
    "节奏": "争议 炎上 骂战 喷 差评 负面",
    "入坑": "推荐 好玩 值得玩 适合新手",
    "情感": "情绪 态度 看法 满意 不满",
    "动向": "动态 新闻 消息 新出",
    "舆论": "风评 口碑 争议 节奏 负面 讨论",
    "推荐": "好玩的 值得 入坑 安利 必玩",
}


def expand_query(query: str) -> str:
    tokens = list(jieba.cut(query))
    expanded = list(tokens)
    for token in tokens:
        if token in SYNONYM_MAP:
            expanded.append(SYNONYM_MAP[token])
    return " ".join(expanded)


def _tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


# ── BM25 post-filter: match ChromaDB WHERE conditions ──

def _match_meta_filters(metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """Check if a single doc's metadata matches a ChromaDB-style WHERE clause.
    Supports: $and, $eq, $gte, $lte, $gt, $lt on string and numeric fields."""
    if not where:
        return True

    if "$and" in where:
        return all(_match_meta_filters(metadata, cond) for cond in where["$and"])

    for field, condition in where.items():
        if field.startswith("$"):
            continue
        meta_val = metadata.get(field)

        if isinstance(condition, dict):
            # Numeric/string comparison operators
            if "$gte" in condition:
                cmp_val = condition["$gte"]
                try:
                    if field == "publish_time":
                        if not meta_val or str(meta_val)[:10] < str(cmp_val)[:10]:
                            return False
                    elif meta_val is None or float(meta_val) < float(cmp_val):
                        return False
                except (ValueError, TypeError):
                    if str(meta_val or "") < str(cmp_val or ""):
                        return False
            if "$lte" in condition:
                cmp_val = condition["$lte"]
                try:
                    if field == "publish_time":
                        if not meta_val or str(meta_val)[:10] > str(cmp_val)[:10]:
                            return False
                    elif meta_val is None or float(meta_val) > float(cmp_val):
                        return False
                except (ValueError, TypeError):
                    if str(meta_val or "") > str(cmp_val or ""):
                        return False
            if "$gt" in condition:
                cmp_val = condition["$gt"]
                try:
                    if meta_val is None or float(meta_val) <= float(cmp_val):
                        return False
                except (ValueError, TypeError):
                    if str(meta_val or "") <= str(cmp_val or ""):
                        return False
            if "$lt" in condition:
                cmp_val = condition["$lt"]
                try:
                    if meta_val is None or float(meta_val) >= float(cmp_val):
                        return False
                except (ValueError, TypeError):
                    if str(meta_val or "") >= str(cmp_val or ""):
                        return False
        else:
            # Equality match
            if str(meta_val or "") != str(condition or ""):
                return False

    return True


# ── VectorDBManager ──

class VectorDBManager:

    def __init__(self, chroma_path: str, collection_name: str, embedding_model: str,
                 bm25_enabled: bool = True):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer(embedding_model)

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[str] = []
        self._bm25_ids: List[str] = []
        self._bm25_metadatas: List[Dict[str, Any]] = []
        self._bm25_dirty = False
        self._bm25_lock = threading.Lock()
        self._bm25_loading = False
        self._bm25_ready = False

        logger.info("VectorDB initialized: %s, collection: %s, count: %d",
                    chroma_path, collection_name, self.collection.count())

        if bm25_enabled:
            self._load_bm25_cache()
        else:
            logger.info("BM25 disabled, skipping cache load")

    def reconnect(self):
        """Reopen ChromaDB connection after fork (for gunicorn --preload)."""
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info("VectorDB reconnected after fork")

    def _load_bm25_cache(self):
        cache_path = os.path.join(self.chroma_path, "bm25_cache.pkl")
        if not os.path.exists(cache_path):
            logger.info("No BM25 cache found at %s", cache_path)
            return
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_docs = data["docs"]
            self._bm25_ids = data["ids"]
            self._bm25_metadatas = data["metadatas"]
            self._bm25 = data["bm25"]
            self._bm25_ready = True
            self._bm25_dirty = False
            logger.info("BM25 loaded from cache: %d docs", len(self._bm25_docs))
        except Exception as e:
            logger.warning("Failed to load BM25 cache: %s", str(e))

    # ── BM25 management ──

    def _load_bm25_data(self):
        """Quick: load docs/metadata from ChromaDB. Does NOT build BM25 index."""
        try:
            total = self.collection.count()
            batch = self.collection.get(include=["documents", "metadatas"], limit=total)
            self._bm25_docs = list(batch.get("documents") or [])
            self._bm25_ids = list(batch.get("ids") or [])
            self._bm25_metadatas = list(batch.get("metadatas") or [])
            logger.info("BM25 data loaded: %d docs", len(self._bm25_docs))
        except Exception as e:
            logger.warning("BM25 data load skipped: %s", str(e))

    def _save_bm25_cache(self):
        cache_path = os.path.join(self.chroma_path, "bm25_cache.pkl")
        try:
            data = {
                "docs": self._bm25_docs,
                "ids": self._bm25_ids,
                "metadatas": self._bm25_metadatas,
                "bm25": self._bm25,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("BM25 cache saved: %d docs, %.1fMB", len(self._bm25_docs),
                       os.path.getsize(cache_path) / 1024 / 1024)
        except Exception as e:
            logger.warning("BM25 cache save failed: %s", str(e))

    def _build_bm25_index(self):
        """Slow: tokenize all docs and build BM25Okapi. Runs in background thread."""
        logger.info("BM25 index build started (%d docs)...", len(self._bm25_docs))
        try:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._bm25_docs])
            self._bm25_ready = True
            self._bm25_dirty = False
            logger.info("BM25 index build complete (%d docs)", len(self._bm25_docs))
            self._save_bm25_cache()
        except Exception as e:
            logger.error("BM25 index build failed: %s", str(e))
        finally:
            self._bm25_loading = False

    def _build_bm25_index_sync(self):
        """Build BM25Okapi synchronously (for offline cache building)."""
        if not self._bm25_docs:
            self._load_bm25_data()
        if self._bm25_docs:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._bm25_docs])
            self._bm25_ready = True
            self._bm25_dirty = False
            self._save_bm25_cache()

    def _ensure_bm25(self):
        """Thread-safe BM25 init. Tries cache first, else background build.
        Returns True if BM25 is ready for searching."""
        if self._bm25_ready:
            return True
        with self._bm25_lock:
            if self._bm25_ready:
                return True
            # Try cache first (instant load if pre-built)
            if not self._bm25_docs and not self._bm25_loading:
                self._load_bm25_cache()
            # Fall back to ChromaDB data + background build
            if not self._bm25_docs and not self._bm25_loading:
                self._load_bm25_data()
            if self._bm25_docs and not self._bm25_loading:
                self._bm25_loading = True
                t = threading.Thread(target=self._build_bm25_index, daemon=True)
                t.start()
                logger.info("BM25 background build started")
        return False

    # ── Score helpers ──

    def _time_decay(self, publish_time: str, half_life_days: int = 30) -> float:
        if not publish_time:
            return 0.5
        try:
            pub = datetime.strptime(publish_time[:10], "%Y-%m-%d")
            days = (datetime.now() - pub).days
            if days <= 0:
                return 1.0
            return 0.5 ** (days / half_life_days)
        except Exception:
            return 0.5

    def _hot_boost(self, metadata: Dict[str, Any]) -> float:
        hot = metadata.get("hot_score", 0) or 0
        if hot <= 0:
            return 1.0
        return 1.0 + min(0.25, math.log10(float(hot) + 0.01) * 0.08)

    # ── Embedding ──

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()

    # ── Search ──

    def search_dense(self, query: str, top_k: int = 20,
                     where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 / (1.0 + results["distances"][0][i]),
                "id": results.get("ids", [[""]])[0][i]
            })
        return formatted

    def search_by_keyword(self, query: str, keyword: str, top_k: int = 20,
                          fetch_k: int = 500) -> List[Dict[str, Any]]:
        """Pre-filter by keyword via metadata, then rank by embedding similarity.
        Works around ChromaDB WHERE being post-ANN filter (not pre-filter).
        Uses collection.get() for pure metadata filtering, then re-ranks."""
        batch = self.collection.get(
            where={"keyword": keyword},
            limit=fetch_k,
            include=["documents", "metadatas", "embeddings"],
        )
        if not batch or not batch.get("ids"):
            return []

        query_emb = self.embed_text(query)
        embeds = batch.get("embeddings")
        try:
            if embeds is None or len(embeds) == 0:
                return []
        except Exception:
            return []

        import numpy as np
        q = np.array(query_emb)
        q_norm = np.linalg.norm(q) or 1.0
        results = []
        for i, doc_emb in enumerate(embeds):
            d = np.array(doc_emb)
            d_norm = np.linalg.norm(d) or 1.0
            sim = float(np.dot(q, d) / (q_norm * d_norm))
            results.append({
                "document": batch["documents"][i] if batch.get("documents") else "",
                "metadata": batch["metadatas"][i] if batch.get("metadatas") else {},
                "score": sim,
                "id": batch["ids"][i],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_bm25(self, query: str, top_k: int = 20,
                    where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._ensure_bm25():
            return []  # BM25 not ready yet, fall back to dense-only
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Fetch more candidates if we need to post-filter
        fetch_k = top_k * 5 if where else top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            metadata = self._bm25_metadatas[idx] if idx < len(self._bm25_metadatas) else {}
            if not metadata:
                doc_id = self._bm25_ids[idx] if idx < len(self._bm25_ids) else None
                if doc_id:
                    try:
                        r = self.collection.get(ids=[doc_id], include=["metadatas"])
                        if r and r.get("metadatas"):
                            metadata = r["metadatas"][0] or {}
                    except Exception:
                        pass
            # Post-filter by WHERE
            if where and not _match_meta_filters(metadata, where):
                continue

            results.append({
                "document": self._bm25_docs[idx],
                "metadata": metadata,
                "score": float(score),
                "id": self._bm25_ids[idx] if idx < len(self._bm25_ids) else ""
            })
        return results

    def hybrid_search(self, query: str, top_k: int = 20, rrf_k: int = 60,
                      where: Optional[Dict[str, Any]] = None,
                      time_decay_enabled: bool = True,
                      hot_boost_enabled: bool = True,
                      diversity_enabled: bool = True,
                      diversity_max_per_game: int = 3,
                      bm25_query_override: Optional[str] = None,
                      query_expansion_enabled: bool = True) -> List[Dict[str, Any]]:
        # 1. Dense: ChromaDB native WHERE filter
        dense_results = self.search_dense(query, top_k, where)

        # 2. BM25: post-filter by WHERE (BM25 index is in-memory, can't pre-filter)
        bm25_query = bm25_query_override if bm25_query_override else query
        if query_expansion_enabled:
            bm25_query = expand_query(bm25_query)
        bm25_results = self.search_bm25(bm25_query, top_k, where)

        # 3. RRF fusion
        score_map: Dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(dense_results, start=1):
            key = item["id"] if item["id"] else item["document"]
            score_map[key] = {"item": item, "rrf": 1.0 / (rrf_k + rank)}

        for rank, item in enumerate(bm25_results, start=1):
            key = item["id"] if item["id"] else item["document"]
            rrf_score = 1.0 / (rrf_k + rank)
            if key in score_map:
                score_map[key]["rrf"] += rrf_score
            else:
                score_map[key] = {"item": item, "rrf": rrf_score}

        # 4. Score multipliers
        for entry in score_map.values():
            meta = entry["item"].get("metadata", {})
            multiplier = 1.0

            if time_decay_enabled:
                decay = self._time_decay(meta.get("publish_time", ""))
                multiplier *= (0.5 + 0.5 * decay)

            if hot_boost_enabled:
                multiplier *= self._hot_boost(meta)

            entry["rrf"] *= multiplier

        # 5. Sort + diversity cap
        merged = sorted(score_map.values(), key=lambda x: x["rrf"], reverse=True)

        if diversity_enabled:
            selected = []
            counts: Dict[str, int] = {}
            for entry in merged:
                kw = entry["item"].get("metadata", {}).get("keyword", "") or "__"
                if counts.get(kw, 0) >= diversity_max_per_game:
                    continue
                selected.append(entry)
                counts[kw] = counts.get(kw, 0) + 1
                if len(selected) >= top_k:
                    break
            return [m["item"] for m in selected]

        return [m["item"] for m in merged[:top_k]]

    # ── Write ──

    def add_documents(self, documents, metadatas, ids=None):
        embeddings = self.embed_batch(documents)
        self.collection.add(documents=documents, embeddings=embeddings,
                           metadatas=metadatas, ids=ids)
        if ids is None:
            ids = [str(len(self._bm25_docs) + i) for i in range(len(documents))]
        self._bm25_docs.extend(documents)
        self._bm25_ids.extend(ids)
        self._bm25_metadatas.extend(metadatas)
        self._bm25_dirty = True

    def rebuild_bm25(self):
        self._ensure_bm25()
        logger.info("BM25 rebuilt: %d docs", len(self._bm25_docs))

    def count(self) -> int:
        return self.collection.count()

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "chroma_path": self.chroma_path,
            "bm25_docs": len(self._bm25_docs),
            "bm25_ready": self._bm25_ready,
        }
