r"""Read-only inspection of the existing Pinecone index.

This script never writes to Pinecone.  It helps you discover the index
dimension and metric, the namespaces that contain vectors, and the metadata
field that holds chunk text.

Usage (from the project root, with ``.env`` populated):

    .\.venv\Scripts\python.exe scripts\inspect_index.py
    .\.venv\Scripts\python.exe scripts\inspect_index.py --sample 3 --preview 160

``--sample`` fetches a few vector IDs and lists their metadata keys so you can
set ``PINECONE_TEXT_FIELD`` correctly.  ``--preview`` additionally prints a
short, single-line preview of the configured text field (off by default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class InspectSettings(BaseSettings):
    """Only the Pinecone settings needed for inspection (Gemini is not required)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    pinecone_api_key: SecretStr
    pinecone_index_name: str
    pinecone_host: str | None = None
    pinecone_namespace: str = ""
    pinecone_embed_model: str = "multilingual-e5-large"
    pinecone_text_field: str = "text"
    embedding_dimension: int | None = Field(default=None)
    expected_metric: str | None = Field(default=None)


def _normalise_host(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().rstrip("/")
    if not value:
        return None
    if "://" not in value:
        value = f"https://{value}"
    if not value.lower().startswith("https://"):
        raise SystemExit("PINECONE_HOST must use https://")
    return value


def _normalise_namespace(value: str) -> str:
    value = value.strip()
    return "" if value in ("", "__default__") else value


def _namespace_count(namespaces: Any, target: str) -> int | None:
    if not namespaces:
        return None
    if target == "":
        for key in ("", "__default__"):
            if key in namespaces:
                return _count(namespaces[key])
    if target in namespaces:
        return _count(namespaces[target])
    return None


def _count(summary: Any) -> int:
    if isinstance(summary, dict):
        return int(summary.get("vector_count", 0))
    return int(getattr(summary, "vector_count", 0))


def _preview(text: str, limit: int) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[:limit].rstrip() + "..."


def inspect(*, sample: int, preview_chars: int) -> int:
    from pinecone import Pinecone

    settings = InspectSettings()  # type: ignore[call-arg]
    host = _normalise_host(settings.pinecone_host)
    namespace = _normalise_namespace(settings.pinecone_namespace)
    index_name = settings.pinecone_index_name.strip()

    if not index_name:
        raise SystemExit("PINECONE_INDEX_NAME must not be blank.")

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value(), timeout=20)
    try:
        index = pc.Index(host=host) if host else pc.Index(name=index_name)
        stats = index.describe_index_stats()

        dimension = getattr(stats, "dimension", None)
        metric = getattr(stats, "metric", None)
        total = getattr(stats, "total_vector_count", None)
        namespaces = getattr(stats, "namespaces", None) or {}

        print(f"Index:        {index_name}")
        print(f"Host:         {host or '(resolved from index name)'}")
        print(f"Dimension:    {dimension if dimension is not None else 'unknown'}")
        print(f"Metric:       {metric if metric is not None else 'unknown'}")
        print(f"Total vectors: {total if total is not None else 'unknown'}")
        print(f"Namespace:    {namespace!r}")

        print("\nNamespaces:")
        if not namespaces:
            print("  (none reported)")
        else:
            for name, summary in namespaces.items():
                label = name if name != "" else "(default)"
                print(f"  {label!r}: {_count(summary)} vector(s)")

        problems: list[str] = []
        count = _namespace_count(namespaces, namespace)
        if total == 0:
            problems.append("The index contains no vectors at all.")
        elif count is None:
            problems.append(
                f"Namespace {namespace!r} was not found. Choose one of the namespaces above."
            )
        elif count == 0:
            problems.append(f"Namespace {namespace!r} exists but contains no vectors.")

        if (
            settings.embedding_dimension is not None
            and dimension is not None
            and dimension != settings.embedding_dimension
        ):
            problems.append(
                f"EMBEDDING_DIMENSION={settings.embedding_dimension} does not match the index "
                f"dimension {dimension}. Set it to {dimension} only if the ingestion model "
                "really produced that dimension."
            )

        if (
            settings.expected_metric
            and metric
            and settings.expected_metric.lower() != str(metric).lower()
        ):
            problems.append(
                f"EXPECTED_METRIC={settings.expected_metric!r} does not match the index metric "
                f"{str(metric)!r}."
            )

        print("\nChecks:")
        if problems:
            for problem in problems:
                print(f"  WARN  {problem}")
        else:
            print("  OK    Dimension, metric, and namespace are consistent with .env.")

        # A model name is not recorded in index stats. This is a reminder, not
        # something the script can verify.
        print(
            "  NOTE  Pinecone does not report the embedding model used at ingestion. "
            f"Confirm PINECONE_EMBED_MODEL={settings.pinecone_embed_model!r} matches it."
        )

        if sample > 0:
            print(f"\nSampling up to {sample} vector(s) from namespace {namespace!r}...")
            ids: list[str] = []
            for page in index.list(namespace=namespace, limit=min(sample, 100)):
                for item in getattr(page, "vectors", None) or []:
                    item_id = getattr(item, "id", None)
                    if item_id is None and isinstance(item, dict):
                        item_id = item.get("id")
                    if item_id:
                        ids.append(str(item_id))
                    if len(ids) >= sample:
                        break
                if len(ids) >= sample:
                    break

            if not ids:
                print("  No vector IDs were returned; the namespace may be empty.")
            else:
                response = index.fetch(ids=ids, namespace=namespace)
                vectors = getattr(response, "vectors", None) or {}
                print(f"  Fetched {len(vectors)} vector(s).")
                for vector_id, vector in vectors.items():
                    metadata = getattr(vector, "metadata", None)
                    if metadata is None and isinstance(vector, dict):
                        metadata = vector.get("metadata")
                    metadata = metadata or {}
                    keys = sorted(metadata.keys())
                    print(f"  - {vector_id}: metadata keys = {keys}")
                    value = metadata.get(settings.pinecone_text_field)
                    if isinstance(value, str):
                        print(
                            f"    {settings.pinecone_text_field!r}: present, "
                            f"{len(value)} character(s)"
                        )
                        if preview_chars > 0:
                            print(f"    preview: {_preview(value, preview_chars)}")
                    else:
                        print(
                            f"    {settings.pinecone_text_field!r}: MISSING or not a string. "
                            "Set PINECONE_TEXT_FIELD to one of the keys above."
                        )

        print("\nNo writes were performed.")
        return 0 if not problems else 2
    finally:
        close = getattr(pc, "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="fetch up to N vector IDs and list their metadata keys (read-only)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="CHARS",
        help="with --sample, print a short preview of the text field (default: off)",
    )
    args = parser.parse_args()
    if args.sample < 0 or args.preview < 0:
        parser.error("--sample and --preview must be non-negative")
    try:
        return inspect(sample=args.sample, preview_chars=args.preview)
    except Exception as exc:
        print(f"Inspection failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        print("Check .env, network access, and that the API key can read this index.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
