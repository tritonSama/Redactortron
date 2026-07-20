"""Optional FastAPI HTTP API for Redactortron.

Install extras::

    pip install -e ".[api]"

Then::

    redactortron serve
    # GET  /health
    # POST /v1/scan
    # POST /v1/redact
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from redactortron import __version__
from redactortron.exceptions import RedactortronError
from redactortron.service import RedactortronService

logger = logging.getLogger("redactortron.api")

_service: Optional[RedactortronService] = None


def get_service() -> RedactortronService:
    global _service
    if _service is None:
        _service = RedactortronService()
    return _service


def create_app(service: Optional[RedactortronService] = None):
    """Build a FastAPI application bound to *service* (or a shared default)."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the HTTP API. Install with: pip install -e '.[api]'"
        ) from exc

    app = FastAPI(
        title="Redactortron",
        version=__version__,
        description=(
            "Local AI-powered document redaction API "
            "(docTR OCR + GLiNER + OpenCV blur)."
        ),
    )
    svc = service or get_service()

    @app.exception_handler(RedactortronError)
    async def _handle_redactortron_error(_request, exc: RedactortronError):
        # Log internal (fingerprinted) event; return public-safe JSON only.
        exc.log(audience="internal")
        return JSONResponse(status_code=400, content=exc.to_dict(audience="public"))

    @app.get("/health")
    def health():
        return svc.health()

    @app.post("/v1/scan")
    async def scan(
        file: UploadFile = File(..., description="PDF or image to scan"),
        threshold: float = Form(0.4),
    ):
        suffix = Path(file.filename or "upload.bin").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        try:
            summary, _result = svc.scan(tmp_path, threshold=threshold)
            return summary.to_dict()
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/v1/redact")
    async def redact(
        file: UploadFile = File(..., description="PDF or image to redact"),
        categories: str = Form(
            ...,
            description="Comma-separated categories, e.g. PERSON,EMAIL",
        ),
        threshold: float = Form(0.4),
    ):
        selected: List[str] = [c.strip() for c in categories.split(",") if c.strip()]
        suffix = Path(file.filename or "upload.bin").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(raw)
            src = Path(tmp.name)

        out_suffix = ".pdf" if suffix.lower() == ".pdf" else suffix
        out = Path(
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=out_suffix,
                prefix="redacted_",
            ).name
        )

        try:
            written = svc.redact(
                src,
                categories=selected,
                output=out,
                threshold=threshold,
            )
            return FileResponse(
                path=str(written),
                filename=f"redacted{out_suffix}",
                media_type="application/octet-stream",
            )
        finally:
            src.unlink(missing_ok=True)

    return app


def launch(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to serve the API. Install with: pip install -e '.[api]'"
        ) from exc

    logger.info("Starting Redactortron API on http://%s:%s", host, port)
    uvicorn.run(
        "redactortron.api:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    launch()


if __name__ == "__main__":
    main()
