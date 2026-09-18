from __future__ import annotations

from typing import Any


class SentenceEmbeddingScorer:
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        instruction: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.instruction = instruction or (
            "Given a sentence completion, retrieve semantically equivalent acceptable "
            "completions."
        )
        self._model: Any | None = None

    def __call__(self, generated: str, references: list[str]) -> dict[str, Any]:
        if not generated.strip():
            raise ValueError("cannot embed an empty generated completion")
        if not references:
            raise ValueError("sentence similarity requires at least one reference")
        model = self._load_model()
        query = f"Instruct: {self.instruction}\nQuery: {generated.strip()}"
        embeddings = model.encode(
            [query, *references],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        similarities = (embeddings[0] @ embeddings[1:].T).detach().float().cpu().tolist()
        scores = [float(value) for value in similarities]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return {
            "embedding_similarity": scores[best_index],
            "embedding_similarities": scores,
            "best_embedding_reference_index": best_index,
            "embedding_model": self.model_id,
            "embedding_model_revision": self.revision,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-completion evaluation requires the 'quality' dependency extra"
                ) from exc
            kwargs = {"revision": self.revision} if self.revision else {}
            self._model = SentenceTransformer(self.model_id, **kwargs)
        return self._model
