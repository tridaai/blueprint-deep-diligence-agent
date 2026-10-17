"""Runtime configuration, from the environment.

Profiles:
  * Local (default): SQLite case store + local object directory. No keys.
  * Production: DATABASE_URL points at PostgreSQL and S3_* at an
    S3-compatible object store (SeaweedFS in the compose topology).
  * Model: a real provider when one is configured (LLM_BACKEND /
    provider keys), otherwise the labelled offline stub model.
  * Harness profile: PROFILE_ID selects the profile artifact from
    profiles/ (default offline-stub); PROFILE_DIR overrides where
    artifacts live. The profile is configuration, independent of
    which model is injected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    database_url: str | None = None
    sqlite_path: str = "./data/diligence.db"
    migrations_dir: str = "migrations"
    # Object storage: S3 when s3_endpoint_url is set, local dir otherwise.
    s3_endpoint_url: str | None = None
    s3_bucket: str = "diligence-evidence"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "us-east-1"
    object_dir: str = "./data/objects"
    # Model provider boundary.
    llm_backend: str = "offline"  # offline | openai | anthropic | ollama
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5.1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"
    api_key: str | None = None
    # Harness profile selection (profiles.py). The profile is loaded
    # and validated at service startup; a bad profile refuses startup.
    profile_id: str = "offline-stub"
    profile_dir: str | None = None
    # Cost accounting: PRICING_DIR overrides where the governed
    # price table lives (default: the repo's pricing/ directory).
    # The table is loaded and validated at startup like a profile.
    pricing_dir: str | None = None
    # Source governance: how long a standard-retention case is
    # kept before the retention sweep lists it for an analyst's
    # decision (RETENTION_DAYS). The sweep lists; it never deletes.
    retention_days: int = 365

    @property
    def store_backend(self) -> str:
        return "postgres" if self.database_url else "sqlite"

    @property
    def object_backend(self) -> str:
        return "s3" if self.s3_endpoint_url else "local"


def load_settings(**overrides) -> Settings:
    env = os.environ
    values = dict(
        database_url=env.get("DATABASE_URL") or None,
        sqlite_path=env.get("SQLITE_PATH", "./data/diligence.db"),
        migrations_dir=env.get("MIGRATIONS_DIR", "migrations"),
        s3_endpoint_url=env.get("S3_ENDPOINT_URL") or None,
        s3_bucket=env.get("S3_BUCKET", "diligence-evidence"),
        s3_access_key_id=env.get("S3_ACCESS_KEY_ID") or None,
        s3_secret_access_key=env.get("S3_SECRET_ACCESS_KEY") or None,
        s3_region=env.get("S3_REGION", "us-east-1"),
        object_dir=env.get("OBJECT_DIR", "./data/objects"),
        llm_backend=(env.get("LLM_BACKEND") or "").lower() or "offline",
        openai_api_key=env.get("OPENAI_API_KEY") or None,
        openai_base_url=env.get("OPENAI_BASE_URL") or None,
        openai_model=env.get("OPENAI_MODEL", "gpt-5.1"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
        anthropic_model=env.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=env.get("OLLAMA_MODEL", "llama3.1"),
        api_key=env.get("API_KEY") or None,
        profile_id=env.get("PROFILE_ID") or "offline-stub",
        profile_dir=env.get("PROFILE_DIR") or None,
        pricing_dir=env.get("PRICING_DIR") or None,
        retention_days=int(env.get("RETENTION_DAYS") or 365),
    )
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
