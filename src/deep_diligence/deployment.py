"""Deployment profiles: named, validated runtime compositions.

Settings name the parts — a store, an object store, a model
backend, a harness profile — but a deployment is the composition,
and some compositions are wrong in ways no single setting can
see: a local boot that silently picks up a production database
URL, a hosted boot with no database at all, a live model paired
with a harness profile that never earned production through the
eval gate. A deployment profile is one named composition, chosen
with DEPLOYMENT_PROFILE and validated as a whole: at settings
load for everything settings can judge (`validate_deployment`,
called from `load_settings` whenever a profile is explicitly
selected), and at service startup for the one rule that needs
the store — live permission flows from the promotion log
(`check_live_permission`).

The set is closed, in code:

- ``local`` — the developer and evaluation composition: SQLite
  at SQLITE_PATH, the local object directory, the offline stub
  harness as the default profile, live models forbidden. A
  DATABASE_URL or S3 endpoint in the environment contradicts
  the composition and refuses, rather than silently re-pointing
  the local boot at production stores.
- ``hosted`` — the production composition: PostgreSQL via
  DATABASE_URL (required — there is no default connection
  string), object storage chosen explicitly (OBJECT_DIR or
  S3_ENDPOINT_URL — never the silent local default), the harness
  default is whatever the profile regression gate (R7) has
  promoted, and live models are permitted only through a
  promoted harness profile.
- ``hosted-self`` — hosted, for a self-hosted model endpoint:
  the model is reached through the OpenAI-compatible boundary
  (LLM_BACKEND=openai) at OPENAI_BASE_URL, which is required
  and has no default on purpose — a silent default would send
  diligence traffic to a public API the deployment never chose.
  This is the house pattern for self-hosted endpoints (NVIDIA
  NIM and its kin): base URL + key in the environment, the model
  named by the harness profile that targets it.

Naming a profile is the act that binds its rules. Settings built
without naming one (the test suite's overrides, the eval packs'
scratch compositions) keep the behaviour they always had — the
profiles govern deployments, and a deployment names its profile.
"""

from __future__ import annotations

import os

from .profiles import load_profile

# The closed set, in the order error messages list it. A name
# outside this tuple is not a profile with reduced checks; it is
# a boot refusal.
DEPLOYMENT_PROFILES = ("local", "hosted", "hosted-self")

# Compositions whose live models are permitted, and only through
# a promoted harness profile. `local` is absent on purpose.
_PROMOTED_ONLY = ("hosted", "hosted-self")


class ConfigError(RuntimeError):
    """A deployment composition that refuses to boot. The message
    names the profile and the requirement it failed."""


def _fail(profile: str, problem: str) -> ConfigError:
    return ConfigError(f"deployment profile {profile!r}: {problem}")


# -- the record -------------------------------------------------------------


def unknown_deployment() -> dict:
    """The read for a case or run recorded before deployment
    profiles existed (R19): unknown, and never backfilled with
    the current boot's values — a record says what was true when
    the work ran, and for those runs nobody recorded it."""
    return {"profile": "unknown", "harness_profile": None, "store": "unknown"}


def deployment_view(record) -> dict:
    """A recorded deployment as the reads present it: the record
    itself when one was stamped, the unknown view when the case
    or run predates deployment records."""
    if isinstance(record, dict) and record.get("profile"):
        return record
    return unknown_deployment()


def deployment_record(settings, harness_profile_id: str) -> dict:
    """The stamp a case and each of its runs carry: the deployment
    profile's name, the harness profile the work ran under, and
    the store backend class. Identities and shapes only — never a
    secret, never a URL (a DATABASE_URL can carry credentials)."""
    return {
        "profile": settings.deployment_profile,
        "harness_profile": harness_profile_id,
        "store": settings.store_backend,
    }


# -- validation at settings load ---------------------------------------------


def _validate_local(settings) -> None:
    if settings.database_url:
        raise _fail(
            "local",
            "the local composition stores cases in SQLite at "
            "SQLITE_PATH, but DATABASE_URL is set — unset "
            "DATABASE_URL, or select a hosted deployment profile",
        )
    if settings.s3_endpoint_url:
        raise _fail(
            "local",
            "the local composition keeps evidence in the local "
            "object directory, but S3_ENDPOINT_URL is set — unset "
            "it, or select a hosted deployment profile",
        )
    if settings.llm_backend != "offline":
        raise _fail(
            "local",
            f"live models are forbidden under the local deployment "
            f"profile, but LLM_BACKEND={settings.llm_backend!r} "
            "selects a live model — the local profile runs the "
            "labelled offline stub; a live model belongs to a "
            "hosted deployment, behind a promoted harness profile",
        )
    profile = load_profile(settings.profile_id, settings.profile_dir)
    if not profile.target_model.startswith("offline:"):
        raise _fail(
            "local",
            f"the deployment's default harness profile is "
            f"{profile.id!r}, which targets "
            f"{profile.target_model!r} — a live model. The local "
            "deployment profile's harness default is the offline "
            "stub profile; a live profile is a hosted deployment's "
            "to promote through the eval gate",
        )


def _validate_hosted(settings, environ, *, self_hosted: bool) -> None:
    name = settings.deployment_profile
    if not settings.database_url:
        raise _fail(
            name,
            "DATABASE_URL is required — the hosted composition "
            "stores cases in PostgreSQL, and there is no default "
            "connection string to fall back to",
        )
    objects_configured = (
        settings.s3_endpoint_url is not None
        or "OBJECT_DIR" in environ
        or settings.object_dir
        != type(settings).__dataclass_fields__["object_dir"].default
    )
    if not objects_configured:
        raise _fail(
            name,
            "object storage must be configured explicitly — set "
            "OBJECT_DIR or S3_ENDPOINT_URL; the hosted composition "
            "does not fall back to the local default directory",
        )
    if self_hosted:
        if settings.llm_backend != "openai":
            raise _fail(
                name,
                "the self-hosted composition reaches its model "
                "through the OpenAI-compatible boundary, but "
                f"LLM_BACKEND={settings.llm_backend!r} — set "
                "LLM_BACKEND=openai with OPENAI_BASE_URL naming "
                "the endpoint",
            )
        if not settings.openai_base_url:
            raise _fail(
                name,
                "OPENAI_BASE_URL is required — it names the "
                "self-hosted endpoint, and no default is provided "
                "on purpose: a silent default would send diligence "
                "traffic to a public API the deployment never chose",
            )


def validate_deployment(settings, environ=None) -> str:
    """The settings-load half of the composition check, called
    from `load_settings` when a deployment profile is explicitly
    selected (DEPLOYMENT_PROFILE in the environment, or the
    deployment_profile override). Every refusal is a ConfigError
    naming the profile and the failed requirement; nothing falls
    back silently. Returns the profile's name."""
    environ = os.environ if environ is None else environ
    name = settings.deployment_profile
    if name not in DEPLOYMENT_PROFILES:
        raise ConfigError(
            f"unknown DEPLOYMENT_PROFILE {name!r}; valid deployment "
            f"profiles are: {', '.join(DEPLOYMENT_PROFILES)}"
        )
    if name == "local":
        _validate_local(settings)
    else:
        _validate_hosted(settings, environ, self_hosted=name == "hosted-self")
    return name


def check_live_permission(settings, *, profile_promoted: bool) -> None:
    """The startup half: live models under a hosted profile are
    permitted only through a promoted harness profile. The
    promotion log lives in the store, so this runs at service
    startup, once the active profile has been resolved from it.
    A bootstrap profile (PROFILE_ID alone) grants no live
    permission, and this check never widens the R7 gate — it only
    ever refuses; promotion is recorded by the gate or not at
    all."""
    name = settings.deployment_profile
    if name not in _PROMOTED_ONLY:
        return
    if settings.llm_backend == "offline":
        return
    if not profile_promoted:
        raise _fail(
            name,
            f"LLM_BACKEND={settings.llm_backend!r} selects a live "
            "model, but live models are permitted only through a "
            "promoted harness profile and no promotion is recorded "
            "— promote a profile through the eval gate, or run "
            "the offline stub until one is",
        )
