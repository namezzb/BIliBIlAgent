import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import langsmith as ls
from langsmith import Client, tracing_context


SECRET_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "cookies",
    "token",
    "secret",
    "password",
}
MAX_STRING_LENGTH = 1000


class LangSmithRuntimeAudit:
    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str | None,
        project_name: str | None,
        endpoint: str | None,
        workspace_id: str | None,
        web_url: str | None,
        app_name: str,
        environment: str,
    ) -> None:
        if not enabled:
            raise RuntimeError("LANGSMITH_TRACING must be set to true.")

        missing_fields = [
            field_name
            for field_name, value in (
                ("LANGSMITH_API_KEY", api_key),
                ("LANGSMITH_PROJECT", project_name),
            )
            if not value
        ]
        if missing_fields:
            joined = ", ".join(missing_fields)
            raise RuntimeError(f"Missing required LangSmith settings: {joined}.")

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_TRACING_V2"] = "true"

        self.project_name = str(project_name)
        self.app_name = app_name
        self.environment = environment
        self.client = Client(
            api_key=api_key,
            api_url=endpoint,
            workspace_id=workspace_id,
            web_url=web_url,
            hide_inputs=self._sanitize_mapping,
            hide_outputs=self._sanitize_mapping,
            hide_metadata=self._sanitize_mapping,
        )

    def close(self) -> None:
        self.client.flush()

    @contextmanager
    def trace_request(
        self,
        *,
        name: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
        tags: list[str] | None = None,
    ) -> Iterator[Any]:
        sanitized_inputs = self.sanitize_payload(inputs)
        sanitized_metadata = self._sanitize_mapping(metadata)
        with ls.trace(
            name=name,
            run_type="chain",
            inputs=sanitized_inputs,
            metadata=sanitized_metadata,
            tags=tags or [],
            client=self.client,
            project_name=self.project_name,
        ) as run:
            with tracing_context(
                project_name=self.project_name,
                enabled=True,
                client=self.client,
                metadata=sanitized_metadata,
                tags=tags or [],
                parent=run,
            ):
                yield run

    @contextmanager
    def trace_span(
        self,
        *,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[Any]:
        with ls.trace(
            name=name,
            run_type=run_type,
            inputs=self.sanitize_payload(inputs),
            metadata=self._sanitize_mapping(metadata or {}),
            tags=tags or [],
            client=self.client,
            project_name=self.project_name,
        ) as run:
            yield run

    def build_reference(
        self,
        *,
        run_id: str,
        trace_run: Any,
        existing_url: str | None = None,
    ) -> dict[str, str | None]:
        return {
            "langsmith_thread_id": run_id,
            "langsmith_thread_url": existing_url or self._get_run_url(trace_run),
        }

    def sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return self._sanitize_mapping(dict(value))
        if isinstance(value, list):
            return [self.sanitize_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self.sanitize_payload(item) for item in value]
        if isinstance(value, str):
            if len(value) <= MAX_STRING_LENGTH:
                return value
            return f"{value[:MAX_STRING_LENGTH]}...<truncated>"
        return value

    def _sanitize_mapping(self, payload: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower()
            if any(secret_key in lowered for secret_key in SECRET_FIELD_NAMES):
                sanitized[key] = "<redacted>"
                continue
            sanitized[key] = self.sanitize_payload(value)
        return sanitized

    def _get_run_url(self, trace_run: Any) -> str | None:
        get_url = getattr(trace_run, "get_url", None)
        if callable(get_url):
            return str(get_url())
        return None


class NoOpRuntimeAudit:
    """Drop-in replacement used when LangSmith is disabled (local dev)."""

    def __init__(self) -> None:
        self.app_name = "BIliBIlAgent"
        self.environment = "development"

    def close(self) -> None:
        pass

    @contextmanager
    def trace_request(self, *, name: str, inputs: dict, metadata: dict, tags=None) -> Iterator[Any]:
        yield _NoOpRun()

    @contextmanager
    def trace_span(self, *, name: str, run_type: str, inputs: dict, metadata=None, tags=None) -> Iterator[Any]:
        yield _NoOpRun()

    def build_reference(self, *, run_id: str, trace_run: Any, existing_url=None) -> dict:
        return {"langsmith_thread_id": run_id, "langsmith_thread_url": None}

    def sanitize_payload(self, value: Any) -> Any:
        return value


class _NoOpRun:
    def end(self, *args, **kwargs) -> None:
        pass

    def add_metadata(self, *args, **kwargs) -> None:
        pass

    def patch(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, name: str):
        """Silently absorb any other method calls LangSmith internals might make."""
        def _noop(*args, **kwargs):
            return None
        return _noop
