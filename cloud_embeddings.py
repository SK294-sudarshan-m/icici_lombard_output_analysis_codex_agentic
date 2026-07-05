"""Cloud-only semantic categorization using the project's configured embedder.

This module intentionally contains no local embedding implementation.  It loads
``settings.med_embedding_model`` and ``settings.aws_region`` from the project's
``config.py`` and calls that model through the existing Bedrock client factory.
AWS credentials are resolved only by boto3's standard provider chain; access keys,
secret keys and session tokens are never accepted, persisted or logged here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class CloudEmbeddingError(RuntimeError):
    """Raised when the configured cloud embedder cannot produce an embedding."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    region: str
    model_id: str
    credential_source: str = "boto3 default credential provider chain"
    local_fallback: bool = False


class ProjectBedrockEmbedder:
    """Invoke only the embedding model defined by the parent project's settings."""

    def __init__(self, project_root: Path) -> None:
        project_root = project_root.resolve()
        root_text = str(project_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        config_logger = logging.getLogger("config")
        previous_level = config_logger.level
        config_logger.setLevel(logging.ERROR)
        try:
            from config import settings
            from pipeline.clients import create_bedrock_client
        except Exception as exc:  # noqa: BLE001 - configuration import is an external boundary
            raise CloudEmbeddingError(
                "Could not load the project embedding configuration/client. "
                "Run the analyzer with the project's Python environment."
            ) from exc
        finally:
            config_logger.setLevel(previous_level)

        model_id = str(settings.med_embedding_model or "").strip()
        region = str(settings.aws_region or "").strip()
        if not model_id or not region:
            raise CloudEmbeddingError(
                "config.py must provide settings.med_embedding_model and settings.aws_region"
            )
        if "embed" not in model_id.lower():
            raise CloudEmbeddingError(
                f"Configured med_embedding_model does not identify an embedding model: {model_id}"
            )

        self.config = EmbeddingConfig(
            provider="AWS Bedrock",
            region=region,
            model_id=model_id,
        )
        self._client = create_bedrock_client(region)
        self.request_count = 0

    def embed(self, text: str) -> list[float]:
        clean = " ".join(str(text or "").split()).strip()
        if not clean:
            raise CloudEmbeddingError("Refusing to embed empty evidence text")
        # Keep requests bounded well below Titan Text Embeddings v2's input limit.
        clean = clean[:12000]
        try:
            response = self._client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps({"inputText": clean}),
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(response["body"].read())
            vector = payload.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError("response did not contain a non-empty embedding")
            result = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in result):
                raise ValueError("embedding contained non-finite values")
            self.request_count += 1
            return result
        except Exception as exc:  # noqa: BLE001 - fail closed across SDK/model failures
            raise CloudEmbeddingError(
                "The configured Bedrock embedding call failed. No local embedding fallback was used. "
                "Provide a valid project AWS role/profile/environment and retry."
            ) from exc


class EmbeddingCache:
    """Model-scoped cache containing only vectors returned by the cloud model."""

    def __init__(self, path: Path, config: EmbeddingConfig) -> None:
        self.path = path
        self.config = config
        self.items: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if (
                    data.get("model_id") == config.model_id
                    and data.get("region") == config.region
                    and isinstance(data.get("items"), dict)
                ):
                    self.items = data["items"]
            except (OSError, UnicodeError, json.JSONDecodeError):
                self.items = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_or_embed(self, text: str, embedder: ProjectBedrockEmbedder) -> list[float]:
        key = self._key(text)
        cached = self.items.get(key)
        if isinstance(cached, list) and cached:
            self.hits += 1
            return [float(value) for value in cached]
        vector = embedder.embed(text)
        self.items[key] = vector
        self.misses += 1
        return vector

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "provider": self.config.provider,
                    "region": self.config.region,
                    "model_id": self.config.model_id,
                    "local_fallback": False,
                    "items": self.items,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = list(left)
    b = list(right)
    if len(a) != len(b) or not a:
        raise CloudEmbeddingError("Embedding dimensions did not match")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise CloudEmbeddingError("Embedding had zero magnitude")
    return dot / (norm_a * norm_b)


FAILURE_TAXONOMY = {
    "missing_line_items": (
        "The printed bill total exists but OCR or table extraction produced zero or too few charge lines; "
        "line items were lost before consolidation."
    ),
    "returns_or_credit_sign": (
        "Negative returns or credit lines were dropped, assigned the wrong sign, or excluded from the "
        "gross-to-net reconciliation."
    ),
    "final_bill_classification": (
        "No authoritative final bill was classified; procedure, package, deposit, receipt or supporting pages "
        "were used with the wrong document finality."
    ),
    "summary_detail_scope": (
        "Summary headers and detailed lines were duplicated or a detailed final bill with incomplete scope was "
        "preferred over the correct summary control total."
    ),
    "supporting_bill_aggregation": (
        "A final-bill category header did not match the correctly scoped total of pharmacy, investigation or "
        "other supporting bills."
    ),
    "net_component_extraction": (
        "The printed net was lower than gross because discount, deduction, return or payer-share components were "
        "not extracted, so the integrity guard treated the net as unexplained."
    ),
    "confidence_boundary": (
        "Financial values reconcile, but the strict confidence rule rejects a score at or below the threshold, "
        "including an exact boundary score."
    ),
}


def semantic_category(
    evidence_text: str,
    *,
    embedder: ProjectBedrockEmbedder,
    cache: EmbeddingCache,
    taxonomy_vectors: dict[str, list[float]],
) -> dict[str, Any]:
    vector = cache.get_or_embed(evidence_text, embedder)
    scores = {
        label: cosine_similarity(vector, tax_vector)
        for label, tax_vector in taxonomy_vectors.items()
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    return {
        "category": label,
        "similarity": round(score, 4),
        "model_id": embedder.config.model_id,
        "region": embedder.config.region,
        "provider": embedder.config.provider,
        "local_fallback": False,
    }
