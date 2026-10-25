import hashlib
import re
from typing import List
import numpy as np

class Embedder:
    """
    Level 2: Text -> vector, for meaning-based recall.

    Default backend is a hashed character n-gram model (1-3 grams for CJK, words for
    Latin text). It needs no model download and works for Chinese, which a bag of
    whitespace tokens would not. If `sentence-transformers` is installed and a model
    name is given (e.g. "BAAI/bge-m3"), that model is used instead.
    """
    def __init__(self, dim: int = 1024, model_name: str = None):
        self.dim = dim
        self.model = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.dim = self.model.get_sentence_embedding_dimension()
            except Exception:
                self.model = None

    def _features(self, text: str) -> List[str]:
        text = text.lower()
        feats = []
        for word in re.findall(r"[a-z0-9]+", text):
            feats.append("w:" + word)
        cjk = re.sub(r"[^一-鿿]", "", text)
        for n in (1, 2, 3):
            for i in range(len(cjk) - n + 1):
                feats.append(f"{n}:{cjk[i:i + n]}")
        return feats

    def embed(self, text: str) -> np.ndarray:
        if self.model is not None:
            v = np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)
            return v
        v = np.zeros(self.dim, dtype=np.float32)
        for f in self._features(text):
            h = int.from_bytes(hashlib.md5(f.encode("utf-8")).digest()[:8], "little")
            # Longer n-grams carry more meaning than single characters
            weight = 1.0 if f.startswith("1:") else 2.0
            v[h % self.dim] += weight if (h >> 63) & 1 else -weight
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @staticmethod
    def to_bytes(v: np.ndarray) -> bytes:
        return np.asarray(v, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32)
