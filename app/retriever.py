# app/retriever.py
from __future__ import annotations
from pathlib import Path
import os

from typing import List, Dict, Any, Tuple, Optional
import asyncio
import json

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

import faiss  

from .schemas import Message
from .config import get_settings

ScoredMessage = Tuple[Message, str, float]


class MessageStore:
    """
    Message store with FAISS-based semantic search and disk persistence.

    - If FAISS index + metadata exist on disk, load them.
    - Otherwise:
        * fetch from /messages API
        * embed with SentenceTransformers
        * build FAISS index (flat or IVF-PQ)
        * persist index + metadata to disk
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        # Don't load the model at startup – defer until first query
        self._model: Optional[SentenceTransformer] = None

        # In-memory collections
        self._messages: List[Message] = []
        self._texts: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, d)
        self._member_names: List[str] = []

        # FAISS index
        self._index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None

        self._lock = asyncio.Lock()

    def _ensure_model(self) -> None:
        """Load the embedding model on first use."""
        if self._model is None:
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @property
    def member_names(self) -> List[str]:
        return self._member_names

    def is_ready(self) -> bool:
        # We only need an index + texts for search.
        return self._index is not None and len(self._texts) > 0

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------


    async def refresh(self) -> None:
        """
        Refresh from disk if possible; otherwise build from the remote /messages API.
        """
        async with self._lock:
            # Try disk first
            loaded = self._try_load_from_disk()
            if loaded:
                print("[MessageStore] Loaded index + metadata from disk.")
                return

            print("[MessageStore] No persisted index found. Fetching from API and building index...")

            try:
                raw = await self._fetch_messages_raw()
            except Exception as e:
                print(f"[MessageStore] ERROR fetching messages from API: {e!r}")
                # Don't crash the app – just start with an empty store
                self._messages = []
                self._texts = []
                self._embeddings = None
                self._member_names = []
                self._index = None
                return

            print(f"[MessageStore] fetched {len(raw)} raw messages")

            if not raw:
                print("[MessageStore] no texts found, store will be empty")
                self._messages = []
                self._texts = []
                self._embeddings = None
                self._member_names = []
                self._index = None
                return

            self._build_from_raw(raw)
            self._persist_to_disk()
            print("[MessageStore] Index built and persisted to disk.")

    def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        member_name_filter: Optional[str] = None,
    ) -> List[ScoredMessage]:
        """
        Return top-k (Message, text_repr, similarity) tuples using FAISS.

        If store is not ready, we just return [].
        """
        if not self.is_ready():
            print("[MessageStore] semantic_search called but store not ready")
            return []

        k = top_k or self.settings.top_k
        k = min(k, len(self._texts))

        self._ensure_model()

        # Encode query
        q_vec = self._model.encode([query], convert_to_numpy=True).astype("float32")  # (1, d)

        # Search with FAISS
        assert self._index is not None
        if hasattr(self._index, "nprobe"):  # IVF-based index
            self._index.nprobe = self.settings.faiss_nprobe

        distances, indices = self._index.search(q_vec, k)  # (1, k)
        idxs = indices[0]
        dists = distances[0]

        # FAISS returns L2 distances by default. Convert to similarity-ish score:
        # score = 1 / (1 + dist)
        results: List[ScoredMessage] = []
        for idx, dist in zip(idxs, dists):
            if idx < 0:
                continue  # invalid
            msg = self._messages[idx]
            text = self._texts[idx]
            score = 1.0 / (1.0 + float(dist))

            if member_name_filter and msg.member_name:
                if msg.member_name.lower() != member_name_filter.lower():
                    continue

            results.append((msg, text, score))

        return results

    # ------------------------------------------------------------------
    # INTERNAL: build / load / persist
    # ------------------------------------------------------------------

    def _build_from_raw(self, raw: List[Dict[str, Any]]) -> None:
        """
        Build messages, texts, embeddings and FAISS index from raw API data.
        """
        messages: List[Message] = []
        texts: List[str] = []
        member_names: set[str] = set()

        for m in raw:
            msg = Message(
                id=m.get("id"),
                member_id=m.get("member_id") or m.get("user_id"),
                member_name=(
                    m.get("member_name")
                    or m.get("member")
                    or m.get("user_name")
                ),
                text=m.get("text") or m.get("message") or "",
            )
            messages.append(msg)

            text_repr = self._to_text_repr(msg)
            texts.append(text_repr)

            if msg.member_name:
                member_names.add(msg.member_name)

        if not texts:
            self._messages = []
            self._texts = []
            self._embeddings = None
            self._member_names = []
            self._index = None
            return

        # 🔹 make sure the embedding model exists
        self._ensure_model()

        # Compute embeddings
        embeddings = self._model.encode(texts, convert_to_numpy=True).astype("float32")
        print(f"[MessageStore] built embeddings for {len(texts)} texts")

        # Build FAISS index
        index = self._build_faiss_index(embeddings)

        # Update in-memory state
        self._messages = messages
        self._texts = texts
        self._embeddings = embeddings
        self._member_names = sorted(member_names)
        self._index = index
        self._dim = embeddings.shape[1]
        print(f"[MessageStore] member_names count: {len(self._member_names)}")

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index (flat or IVF-PQ) from embeddings.
        """
        settings = self.settings
        n, d = embeddings.shape
        self._dim = d

        index_type = settings.faiss_index_type.lower()

        if index_type == "flat" or n < 1000:
            # For very small datasets, flat index is fine & simple
            print("[MessageStore] Building IndexFlatL2 (exact search)...")
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            return index

        # IVF-PQ (compressed) for larger datasets
        # Heuristic: nlist should not exceed number of vectors
        nlist = min(settings.faiss_nlist, max(1, n // 40))
        m = settings.faiss_m
        print(f"[MessageStore] Building IVF-PQ index (n={n}, d={d}, nlist={nlist}, m={m})...")

        quantizer = faiss.IndexFlatL2(d)
        index_ivfpq = faiss.IndexIVFPQ(
            quantizer,
            d,
            nlist,
            m,
            8,  # bits per subvector
        )

        # Train on the full dataset (or sample if n is huge)
        # For n ~ 1M, it's okay to train on all with MiniLM
        index_ivfpq.train(embeddings)
        index_ivfpq.add(embeddings)

        print("[MessageStore] IVF-PQ index built.")
        return index_ivfpq

    def _to_text_repr(self, msg: Message) -> str:
        """
        Canonical text representation for embedding (includes member name).
        """
        if msg.member_name:
            return f"{msg.member_name}: {msg.text}"
        return msg.text

    def _try_load_from_disk(self) -> bool:
        """
        Try to load FAISS index + metadata from disk.
        Returns True if successful, False otherwise.
        """
        idx_path = Path(self.settings.faiss_index_path).resolve()
        meta_path = Path(self.settings.faiss_meta_path).resolve()

        print(f"[MessageStore] _try_load_from_disk:")
        print(f"  index={idx_path}")
        print(f"  meta={meta_path}")
        print(f"  index exists? {idx_path.is_file()}")
        print(f"  meta exists? {meta_path.is_file()}")

        if not (idx_path.is_file() and meta_path.is_file()):
            print("[MessageStore] index or metadata not found on disk, skipping load.")
            return False

        try:
            print(f"[MessageStore] Loading FAISS index from {idx_path}")
            index = faiss.read_index(str(idx_path))

            print(f"[MessageStore] Loading metadata from {meta_path}")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            messages: list[Message] = []
            texts: list[str] = []
            member_names: set[str] = set()

            for m in meta:
                msg = Message(
                    id=m.get("id"),
                    member_id=m.get("member_id"),
                    member_name=m.get("member_name"),
                    text=m.get("text") or "",
                )
                messages.append(msg)
                text_repr = self._to_text_repr(msg)
                texts.append(text_repr)
                if msg.member_name:
                    member_names.add(msg.member_name)

            self._messages = messages
            self._texts = texts
            self._embeddings = None  # FAISS already has vectors
            self._member_names = sorted(member_names)
            self._index = index
            self._dim = index.d

            print(f"[MessageStore] Loaded {len(self._messages)} messages from disk.")
            return True

        except Exception as e:
            # This is where your current "Failed to load from disk" message comes from
            print(f"[MessageStore] Failed to load from disk: {e!r}")
            return False

    def _persist_to_disk(self) -> None:
        """
        Persist FAISS index + metadata (messages) to disk.
        """
        settings = self.settings
        os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
        os.makedirs(os.path.dirname(settings.faiss_meta_path), exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, settings.faiss_index_path)
            print(f"[MessageStore] FAISS index written to {settings.faiss_index_path}")

        # Save metadata as JSON
        meta: List[Dict[str, Any]] = []
        for m in self._messages:
            meta.append(
                {
                    "id": m.id,
                    "member_id": m.member_id,
                    "member_name": m.member_name,
                    "text": m.text,
                }
            )

        with open(settings.faiss_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        print(f"[MessageStore] metadata written to {settings.faiss_meta_path}")

    # ------------------------------------------------------------------
    # INTERNAL: fetch from external API
    # ------------------------------------------------------------------

    async def _fetch_messages_raw(self) -> List[Dict[str, Any]]:
        """
        Call the external /messages API and normalize to a list of dicts.
        """
        settings = self.settings
        url = str(settings.messages_api_url)

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            # Expected Swagger schema: { "total": int, "items": [ {..}, {..} ] }
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                return data["items"]

            # Fallbacks
            if isinstance(data, list):
                return data

            for v in getattr(data, "values", lambda: [])():
                if isinstance(v, list):
                    return v

            raise RuntimeError(f"Unexpected /messages response format: {data}")


# ----------------------------------------------------------------------
# Global store & helpers
# ----------------------------------------------------------------------

_STORE: Optional[MessageStore] = None
_STORE_LOCK = asyncio.Lock()

async def init_store() -> None:
    """
    Initialize the global MessageStore once.
    """
    global _STORE
    async with _STORE_LOCK:
        if _STORE is not None:
            return

        store = MessageStore()
        try:
            await store.refresh()
            print("[MessageStore] initial refresh succeeded.")
        except Exception as e:
            print(f"[MessageStore] initial refresh failed: {e!r}")
            # Leave _STORE as None so we can detect failure
            return

        _STORE = store
        print("[MessageStore] initialized with",
              len(store._messages), "messages, index ntotal=",
              store._index.ntotal if store._index is not None else 0)

async def get_store() -> "MessageStore":
    """
    Ensure the global store is initialized before use.
    Safe to call from request handlers.
    """
    await init_store()
    if _STORE is None:
        raise RuntimeError("MessageStore is not ready after init_store()")
    return _STORE