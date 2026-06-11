"""Import ChromaDB from exported numpy data — native HNSW rebuild on Linux."""
import os
import sys
import shutil
import logging
import numpy as np
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_export')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'chroma_db')


def main():
    npz_path = os.path.join(INPUT_DIR, 'chroma_data.npz')
    if not os.path.exists(npz_path):
        logger.error("Export file not found: %s", npz_path)
        sys.exit(1)

    logger.info("Loading exported data...")
    data = np.load(npz_path, allow_pickle=True)

    ids = data['ids'].tolist()
    documents = data['documents'].tolist()
    metadatas = data['metadatas'].tolist()
    embeddings = data['embeddings'].tolist()

    logger.info("Loaded: %d ids, %d docs, %d metas, %d embs",
                len(ids), len(documents), len(metadatas), len(embeddings))

    # Remove old corrupted DB if exists
    if os.path.exists(CHROMA_PATH):
        logger.info("Removing old ChromaDB at: %s", CHROMA_PATH)
        shutil.rmtree(CHROMA_PATH)

    logger.info("Creating fresh ChromaDB at: %s", os.path.abspath(CHROMA_PATH))
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.create_collection(name='game_opinions')

    logger.info("Adding %d documents with pre-computed embeddings...", len(documents))
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )
        if (i // batch_size) % 40 == 0:
            logger.info("Progress: %d/%d", batch_end, len(documents))

    logger.info("Import complete. Collection count: %d", collection.count())


if __name__ == '__main__':
    main()
