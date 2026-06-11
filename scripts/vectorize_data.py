"""
Vectorize MySQL data into ChromaDB vector store.
Reads structured opinion data from a MariaDB/MySQL database, builds text documents,
generates embeddings, and stores them in ChromaDB with metadata.

Usage:
    python scripts/vectorize_data.py \\
        --host <DB_HOST> --user <DB_USER> --password <DB_PASSWORD> --database <DB_NAME>
"""
import os
import sys
import logging
import argparse
import pymysql
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from vector_db import VectorDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/vectorize.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PLATFORM_MAP = {1: "B站", 2: "NGA"}


def make_document(row) -> str:
    """Build rich text document for embedding."""
    platform_name = PLATFORM_MAP.get(row["platform"], f"平台{row['platform']}")
    board = row.get("board_name") or "综合"
    type_name = row.get("type") or "帖子"
    keyword = row.get("keyword") or ""
    author = row.get("author") or "未知"
    pub_time = row["publish_time"].strftime("%Y年%m月%d日 %H:%M") if row.get("publish_time") else ""
    title = (row.get("title_clean") or "").strip()
    content = (row.get("content_clean") or "").strip()

    parts = [
        f"这是来自{platform_name}{board}板块的一条{type_name}，讨论{keyword}相关话题。",
        f"作者{author}发布于{pub_time}。",
    ]
    if title:
        parts.append(f"标题：{title}")
    if content:
        parts.append(f"正文：{content}")

    return "".join(parts)


def make_metadata(row) -> dict:
    return {
        "keyword": row.get("keyword") or "",
        "platform": PLATFORM_MAP.get(row["platform"], str(row["platform"])),
        "type": row.get("type") or "",
        "sentiment_score": float(row.get("sentiment_score") or 0),
        "publish_time": row["publish_time"].strftime("%Y-%m-%d %H:%M:%S") if row.get("publish_time") else "",
        "hot_score": float(row.get("hot_score") or 0),
        "author": row.get("author") or "",
        "board_name": row.get("board_name") or "",
    }


def connect_mysql(host, port, user, password, database, unix_socket=None):
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=database, charset='utf8mb4', unix_socket=unix_socket
    )
    logger.info("Connected to MySQL: %s:%d/%s", host, port, database)
    return conn


def fetch_all(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM standardized_data")
    total = cursor.fetchone()[0]
    logger.info("Total records: %d", total)

    cursor.execute("""
        SELECT id, platform, type, author, title_clean, content_clean,
               publish_time, keyword, board_name, sentiment_score, hot_score
        FROM standardized_data
        ORDER BY id
    """)
    return cursor, total


def process(host, port, user, password, database, chroma_path, batch_size=1000, unix_socket=None):
    conn = connect_mysql(host, port, user, password, database, unix_socket=unix_socket)
    try:
        vector_db = VectorDBManager(
            chroma_path=chroma_path,
            collection_name='game_opinions',
            embedding_model='BAAI/bge-small-zh-v1.5'
        )
        cursor, total = fetch_all(conn)
        processed = 0
        batch_docs, batch_metas, batch_ids = [], [], []

        with tqdm(total=total, desc="向量化", unit="条") as pbar:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row_data in rows:
                    row = {
                        "id": row_data[0], "platform": row_data[1], "type": row_data[2],
                        "author": row_data[3], "title_clean": row_data[4],
                        "content_clean": row_data[5], "publish_time": row_data[6],
                        "keyword": row_data[7], "board_name": row_data[8],
                        "sentiment_score": row_data[9], "hot_score": row_data[10],
                    }
                    doc = make_document(row)
                    meta = make_metadata(row)
                    batch_docs.append(doc)
                    batch_metas.append(meta)
                    batch_ids.append(f"doc_{row['id']}")

                    if len(batch_docs) >= batch_size:
                        vector_db.add_documents(batch_docs, batch_metas, batch_ids)
                        processed += len(batch_docs)
                        pbar.update(len(batch_docs))
                        batch_docs, batch_metas, batch_ids = [], [], []

                if batch_docs:
                    vector_db.add_documents(batch_docs, batch_metas, batch_ids)
                    processed += len(batch_docs)
                    pbar.update(len(batch_docs))
                    batch_docs, batch_metas, batch_ids = [], [], []

        logger.info("Vectorization done. Total: %d", processed)
        vector_db.rebuild_bm25()
        info = vector_db.get_collection_info()
        logger.info("Collection: %s", info)
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Vectorize MySQL data to ChromaDB')
    parser.add_argument('--host', help='MySQL host')
    parser.add_argument('--port', type=int, default=3306)
    parser.add_argument('--user', help='MySQL user')
    parser.add_argument('--password', help='MySQL password')
    parser.add_argument('--database', help='MySQL database name')
    parser.add_argument('--chroma-path', default='../chroma_db')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--unix-socket', default=None, help='MySQL unix socket path')
    args = parser.parse_args()

    os.makedirs(args.chroma_path, exist_ok=True)

    logger.info("Starting vectorization...")
    process(
        host=args.host, port=args.port, user=args.user,
        password=args.password, database=args.database,
        chroma_path=args.chroma_path, batch_size=args.batch_size,
        unix_socket=args.unix_socket
    )
    logger.info("Done!")


if __name__ == '__main__':
    main()
