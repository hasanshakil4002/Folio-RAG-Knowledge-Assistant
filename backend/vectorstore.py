"""FAISS vector store.

* The original chunk text lives in chunks.jsonl (the source of truth).
* Each embedding provider+model gets its own FAISS index, because vector dimensions differ.
* A newly uploaded PDF only adds its new chunks to the end of the index; nothing is rebuilt.
"""
import json
import re
import threading
import uuid
from datetime import datetime

import faiss
import numpy as np

from . import config as cfg


class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.chunks: dict[int, dict] = {}
        self.docs: dict[str, dict] = {}
        self.indexes: dict[str, faiss.Index] = {}
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if cfg.CHUNKS_FILE.exists():
            with open(cfg.CHUNKS_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        c = json.loads(line)
                        self.chunks[c["id"]] = c
        if cfg.DOCS_FILE.exists():
            self.docs = json.loads(cfg.DOCS_FILE.read_text(encoding="utf-8"))
        self.next_id = max(self.chunks, default=0) + 1

    def _save_docs(self) -> None:
        cfg.DOCS_FILE.write_text(json.dumps(self.docs, ensure_ascii=False, indent=1),
                                 encoding="utf-8")

    @staticmethod
    def _index_path(key: str):
        return cfg.DATA_DIR / ("index_" + re.sub(r"[^A-Za-z0-9]+", "_", key) + ".faiss")

    def _get_index(self, key: str):
        if key in self.indexes:
            return self.indexes[key]
        path = self._index_path(key)
        if path.exists():
            self.indexes[key] = faiss.read_index(str(path))
            return self.indexes[key]
        return None

    # ---------- documents ----------
    def add_document(self, name: str, items: list[tuple[int, str]], ocr: bool,
                     pages: int, file: str) -> dict:
        with self.lock:
            doc_id = uuid.uuid4().hex[:10]
            recs = []
            for page, text in items:
                recs.append({"id": self.next_id, "doc_id": doc_id, "doc_name": name,
                             "page": page, "text": text})
                self.next_id += 1
            with open(cfg.CHUNKS_FILE, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    self.chunks[r["id"]] = r
            doc = {"id": doc_id, "name": name, "pages": pages, "chunks": len(recs),
                   "ocr": ocr, "file": file,
                   "added": datetime.now().strftime("%Y-%m-%d %H:%M")}
            self.docs[doc_id] = doc
            self._save_docs()
            return doc

    def delete_document(self, doc_id: str) -> bool:
        with self.lock:
            doc = self.docs.pop(doc_id, None)
            if not doc:
                return False
            ids = [i for i, c in self.chunks.items() if c["doc_id"] == doc_id]
            for i in ids:
                del self.chunks[i]
            with open(cfg.CHUNKS_FILE, "w", encoding="utf-8") as f:
                for c in self.chunks.values():
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            self._save_docs()
            if ids:
                arr = np.array(ids, dtype=np.int64)
                for path in cfg.DATA_DIR.glob("index_*.faiss"):
                    idx = faiss.read_index(str(path))
                    idx.remove_ids(arr)
                    faiss.write_index(idx, str(path))
                self.indexes.clear()
            if doc.get("file"):
                (cfg.UPLOAD_DIR / doc["file"]).unlink(missing_ok=True)
            return True

    # ---------- index ----------
    def missing(self, key: str) -> list[int]:
        """Chunks that are not yet in this provider's index."""
        with self.lock:
            idx = self._get_index(key)
            have = set(faiss.vector_to_array(idx.id_map).tolist()) if idx is not None else set()
            return sorted(i for i in self.chunks if i not in have)

    def add_vectors(self, key: str, ids: list[int], vecs: np.ndarray) -> None:
        with self.lock:
            idx = self._get_index(key)
            if idx is None:
                idx = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
                self.indexes[key] = idx
            idx.add_with_ids(vecs, np.array(ids, dtype=np.int64))
            faiss.write_index(idx, str(self._index_path(key)))

    def search(self, key: str, qvec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        with self.lock:
            idx = self._get_index(key)
            if idx is None or idx.ntotal == 0:
                return []
            scores, ids = idx.search(qvec.reshape(1, -1).astype("float32"), k)
            return [(self.chunks[int(i)], float(s))
                    for i, s in zip(ids[0], scores[0]) if int(i) in self.chunks]
