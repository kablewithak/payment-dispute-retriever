from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        *,
        normalize_embeddings: bool = True,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = model or SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0]

    def embedding_dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError("Embedding model returned an invalid dimension.")
        return int(dimension)