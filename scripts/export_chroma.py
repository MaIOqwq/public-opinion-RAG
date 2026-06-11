"""Export ChromaDB collection to portable numpy format — no model needed."""
import os
import numpy as np
import chromadb
from chromadb.config import Settings

CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_export')


def main():
    print(f"Reading ChromaDB from: {os.path.abspath(CHROMA_PATH)}")
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name='game_opinions')
    count = collection.count()
    print(f"Collection count: {count}")

    if count == 0:
        print("ERROR: Collection is empty!")
        return

    print("Fetching all data (this may take a minute)...")
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    embs = data["embeddings"]

    print(f"Retrieved: {len(ids)} ids, {len(docs)} docs, {len(metas)} metas, {len(embs)} embeddings")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    emb_array = np.array(embs, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, 'chroma_data.npz')
    np.savez_compressed(
        out_path,
        ids=np.array(ids, dtype=object),
        documents=np.array(docs, dtype=object),
        metadatas=np.array(metas, dtype=object),
        embeddings=emb_array,
    )

    size_mb = os.path.getsize(out_path) / (1024*1024)
    print(f"Exported to: {out_path}")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
