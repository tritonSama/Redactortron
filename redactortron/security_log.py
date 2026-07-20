"""Secure, structured error serialization (Serilog-style) with PII redaction.

Security goals (aligned with common logging guidance such as OWASP):
- User-facing / API responses must not leak absolute paths, usernames, or
  document titles that may be sensitive.
- Internal logs may keep richer detail but still avoid dumping raw secrets;
  path values are reduced to basename + optional fingerprint.
- Every error gets a short ``correlation_id`` so support/debug can match a
  public message to a private log line without embedding PII in the UI.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

logger = logging.getLogger("redactortron.security")

# Absolute or home-rooted paths embedded in free text.
_ABS_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:[A-Za-z]:[\\/][^\s'\"`,;]+)"  # Windows drive path
    r"|(?:\\\\[^\s'\"`,;]+)"  # UNC
    r"|(?:/(?:Users|home|tmp|var|etc|opt|mnt|private)[^\s'\"`,;]*)"  # Unix-ish
    r"|(?:~[\\/][^\s'\"`,;]+)"  # home-relative
    r")"
)

_SENSITIVE_CONTEXT_KEYS = frozenset(
    {
        "path",
        "source",
        "output",
        "file",
        "filename",
        "filepath",
        "input",
        "upload",
    }
)


def new_correlation_id() -> str:
    """Return a short opaque id safe to show in UIs and tickets."""
    return uuid.uuid4().hex[:12]


def path_fingerprint(path: Union[str, PurePath]) -> str:
    """Stable non-reversible fingerprint for correlating the same file in logs."""
    normalized = str(path).replace("\\", "/").lower().encode("utf-8", errors="replace")
    return hashlib.sha256(normalized).hexdigest()[:12]


def _as_pure_path(value: str) -> PurePath:
    if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\"):
        return PureWindowsPath(value)
    return PurePosixPath(value)


def sanitize_path(value: Any, *, audience: str = "public") -> Any:
    """Redact a filesystem path for the given audience.

    * ``public`` — ``<redacted>/<file>.<ext>`` (no directories, no real filename).
    * ``internal`` — basename + fingerprint (no full directory tree).
    * ``debug`` — original string (local CLI ``-v`` only).
    """
    if value is None or not isinstance(value, (str, PurePath)):
        return value
    text = str(value)
    if audience == "debug":
        return text

    pure = _as_pure_path(text)
    name = pure.name or "document"
    suffix = pure.suffix.lower() if pure.suffix else ""
    if audience == "public":
        # Filenames can be sensitive (e.g. payroll stubs); keep extension only.
        return f"<redacted>/<file>{suffix or ''}"
    return {
        "name": name,
        "fingerprint": path_fingerprint(text),
    }


def sanitize_message(message: str, *, audience: str = "public") -> str:
    """Strip absolute paths from free-text error messages."""
    if audience == "debug":
        return message

    def _replace(match: re.Match[str]) -> str:
        raw = match.group("path")
        sanitized = sanitize_path(raw, audience=audience)
        if isinstance(sanitized, dict):
            return f"<redacted>/{sanitized['name']}#{sanitized['fingerprint']}"
        return str(sanitized)

    return _ABS_PATH_RE.sub(_replace, message)


def sanitize_context(
    context: Mapping[str, Any],
    *,
    audience: str = "public",
) -> Dict[str, Any]:
    """Return a copy of *context* safe for the given audience."""
    cleaned: Dict[str, Any] = {}
    for key, value in context.items():
        key_l = str(key).lower()
        if key_l in _SENSITIVE_CONTEXT_KEYS or key_l.endswith("_path"):
            cleaned[key] = sanitize_path(value, audience=audience)
        elif isinstance(value, str) and _ABS_PATH_RE.search(value):
            cleaned[key] = sanitize_message(value, audience=audience)
        elif isinstance(value, Mapping):
            cleaned[key] = sanitize_context(value, audience=audience)
        else:
            cleaned[key] = value
    return cleaned


def sanitize_cause(
    cause: Optional[BaseException],
    *,
    audience: str = "public",
) -> Optional[Dict[str, str]]:
    """Serialize the underlying exception without leaking pathful messages."""
    if cause is None:
        return None
    name = type(cause).__name__
    if audience == "public":
        # Type only — messages often embed paths (pdf2image, OS errors).
        return {"type": name}
    msg = sanitize_message(str(cause), audience=audience)
    return {"type": name, "message": msg}


def serialize_error(
    *,
    code: str,
    stage: str,
    message: str,
    hint: str,
    context: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
    correlation_id: Optional[str] = None,
    audience: str = "public",
    include_cause: bool = False,
) -> Dict[str, Any]:
    """Build a Serilog-style structured error event.

    The shape mirrors common structured-logging property bags so the same
    payload can be shown in UI, returned from an API, or emitted as JSON logs.
    """
    cid = correlation_id or new_correlation_id()
    event: Dict[str, Any] = {
        "@t": datetime.now(timezone.utc).isoformat(),
        "@l": "Error",
        "@mt": "Redactortron {Code} at stage {Stage}: {Message}",
        "Code": code,
        "Stage": stage,
        "Message": sanitize_message(message, audience=audience),
        "Hint": hint,
        "CorrelationId": cid,
        "Context": sanitize_context(dict(context or {}), audience=audience),
    }
    if include_cause or audience != "public":
        cause_payload = sanitize_cause(
            cause,
            audience="public" if audience == "public" else audience,
        )
        if cause_payload:
            event["Cause"] = cause_payload
    return event


def format_serialized_error(event: Mapping[str, Any], *, verbose: bool = False) -> str:
    """Render a structured error event as a compact multi-line report."""
    lines = [
        f"[{event.get('Code', 'ERROR')}] stage={event.get('Stage', 'unknown')}"
        f" id={event.get('CorrelationId', '-')}",
        f"  Problem: {event.get('Message', '')}",
    ]
    ctx = event.get("Context") or {}
    if ctx:
        detail = ", ".join(f"{k}={v!r}" for k, v in ctx.items())
        lines.append(f"  Where:   {detail}")
    if event.get("Hint"):
        lines.append(f"  Hint:    {event['Hint']}")
    cause = event.get("Cause")
    if verbose and isinstance(cause, Mapping):
        if "message" in cause:
            lines.append(f"  Cause:   {cause.get('type')}: {cause.get('message')}")
        else:
            lines.append(f"  Cause:   {cause.get('type')}")
    return "\n".join(lines)


def emit_structured_log(
    event: Mapping[str, Any],
    *,
    level: int = logging.ERROR,
    extra_properties: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit a JSON-friendly structured log line (Serilog-like property bag)."""
    payload: MutableMapping[str, Any] = dict(event)
    if extra_properties:
        payload.update(extra_properties)
    # Keep message template + properties; avoid interpolating sensitive values
    # into the primary log message beyond code/stage/id.
    logger.log(
        level,
        "Redactortron %s at stage %s (id=%s)",
        payload.get("Code"),
        payload.get("Stage"),
        payload.get("CorrelationId"),
        extra={"serilog": dict(payload)},
    )
