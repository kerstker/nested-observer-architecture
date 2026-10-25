import hashlib
import logging
import os
import re
import threading
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class Embedder:
    """
    Level 2: Text -> vector, for recall by meaning and by wording.

    With the local model (BAAI/bge-small-zh-v1.5 as ONNX, on the CPU with onnxruntime) a vector is
    two halves: what the text *means* (the model) and what it literally *says* (hashed character
    n-grams). Each alone misses things the other catches - the model loses rare names and in-jokes,
    the n-grams lose paraphrases; together they recall clearly better than either (measured on her
    own episodes with scripts/benchmark_memory.py: top-3 recall 53% n-grams, 62% model, 70% both).

    The halves are weighted so that one dot product gives the combined score, and `similarity`
    shifts it so that unrelated text sits near 0 and the same text at 1 - the scale the rest of the
    code was written for. Without the model (no onnxruntime, no model folder, or in tests) vectors
    are n-grams only, as before.

    Vectors from different models live in different spaces and must never be compared: `model_id`
    names the space, and stored vectors carry it (EpisodicMemory.ensure_model re-embeds the rest).
    Stored as float16: half the size, and recall quality is unchanged.
    """
    MODEL_FLOOR = 0.45      # the model's cosine for unrelated text (median of random pairs ~0.44)
    MODEL_WEIGHT = 0.5      # share of the model in the combined score

    def __init__(self, dim: int = 1024, model_dir: str = None, model_name: str = None):
        self.dim = dim
        self.lex_dim = dim
        self.backend = "hash"
        self.model_id = f"hash-{dim}"
        self.offset = 0.0
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()
        if model_dir:
            self._load_onnx(model_dir)
        elif model_name:
            logger.warning("[Embedder] model_name is no longer supported; pass model_dir (an ONNX export)")

    def _load_onnx(self, model_dir: str):
        path = os.path.join(model_dir, "onnx", "model.onnx")
        if not (os.path.exists(path) and os.path.exists(os.path.join(model_dir, "tokenizer.json"))):
            logger.warning(f"[Embedder] no model at {model_dir}; recall by wording only")
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            logger.warning(f"[Embedder] {e}; recall by wording only")
            return
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tok.enable_truncation(256)
        tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2            # leave the CPU to the rest of her
        self._session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tokenizer = tok
        sem_dim = self._session.get_outputs()[0].shape[-1] or 512
        self.backend = "onnx"
        self.lex_dim = 512
        self.dim = sem_dim + self.lex_dim
        f, w = self.MODEL_FLOOR, self.MODEL_WEIGHT
        # score = w * (cos_model - f) / (1 - f) + (1 - w) * cos_ngram  =  a*cos_model + (1-w)*cos_ngram - offset
        self._a = w / (1 - f)
        self.offset = w * f / (1 - f)
        self.model_id = f"{os.path.basename(os.path.normpath(model_dir))}+lex{self.lex_dim}"

    # --- the n-gram half ----------------------------------------------------------

    @staticmethod
    def _features(text: str) -> List[str]:
        text = text.lower()
        feats = []
        for word in re.findall(r"[a-z0-9]+", text):
            feats.append("w:" + word)
        cjk = re.sub(r"[^一-鿿]", "", text)
        for n in (1, 2, 3):
            for i in range(len(cjk) - n + 1):
                feats.append(f"{n}:{cjk[i:i + n]}")
        return feats

    def _hash_embed(self, text: str, dim: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for f in self._features(text):
            h = int.from_bytes(hashlib.md5(f.encode("utf-8")).digest()[:8], "little")
            # Longer n-grams carry more meaning than single characters
            weight = 1.0 if f.startswith("1:") else 2.0
            v[h % dim] += weight if (h >> 63) & 1 else -weight
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    # --- the model half -----------------------------------------------------------

    def _model_embed(self, texts: List[str], batch: int = 32) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch):
            chunk = [t if t.strip() else " " for t in texts[start:start + batch]]
            with self._lock:
                enc = self._tokenizer.encode_batch(chunk)
                ids = np.array([e.ids for e in enc], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
                if "token_type_ids" in self._inputs:
                    feed["token_type_ids"] = np.zeros_like(ids)
                cls = self._session.run(None, feed)[0][:, 0]      # bge: the [CLS] vector is the sentence
            out.append(cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-9))
        return np.concatenate(out).astype(np.float32)

    # --- embedding and comparing --------------------------------------------------

    def embed_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        lex = np.stack([self._hash_embed(t, self.lex_dim) for t in texts])
        if self.backend != "onnx":
            return lex
        sem = self._model_embed(texts)
        return np.concatenate([np.sqrt(self._a) * sem, np.sqrt(1 - self.MODEL_WEIGHT) * lex], axis=1).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def scores(self, matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Similarity of q to every row of matrix, on the shared scale."""
        return matrix @ q - self.offset

    def similarity(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None or len(a) != len(b):
            return 0.0                    # vectors from another model: not comparable
        return float(np.dot(a, b) - self.offset)

    @staticmethod
    def to_bytes(v: np.ndarray) -> bytes:
        return np.asarray(v, dtype=np.float16).tobytes()

    @staticmethod
    def from_bytes(b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float16).astype(np.float32)

def default_model_dir() -> Optional[str]:
    """models/bge-small-zh-v1.5 next to the app (thefirstlife-dev/models), if it is there."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "models", "bge-small-zh-v1.5"))
    return path if os.path.isdir(path) else None
