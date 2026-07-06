"""Standalone Bedrock Runtime client helpers for the analysis agent.

This is intentionally local to ``case_verdict_analysis_agent``.  It does not
import the parent project's ``pipeline.clients`` module, but it keeps the same
practical behavior the project needed on Windows: per-thread clients and
connection-error recovery.
"""

from __future__ import annotations

import logging
import random
import threading
import time

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ConnectionClosedError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    SSLError,
)

try:
    from .standalone_settings import (
        BEDROCK_MAX_POOL_CONNECTIONS,
        BEDROCK_READ_TIMEOUT_SECONDS,
    )
except ImportError:  # Direct script execution.
    from standalone_settings import (
        BEDROCK_MAX_POOL_CONNECTIONS,
        BEDROCK_READ_TIMEOUT_SECONDS,
    )


log = logging.getLogger("case_verdict_analysis_agent.bedrock")

_CONNECTION_ERRORS = (
    SSLError,
    ConnectionClosedError,
    EndpointConnectionError,
    BotoConnectionError,
)
_GUARDED_METHODS = frozenset(
    {"converse", "invoke_model", "invoke_model_with_response_stream"}
)
_MAX_RECONNECTS = 4
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0


class ThreadLocalBedrockRuntimeClient:
    """Per-thread, self-healing Bedrock Runtime client proxy."""

    def __init__(self, region: str) -> None:
        self._region = region
        self._config = BotoConfig(
            read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
            max_pool_connections=BEDROCK_MAX_POOL_CONNECTIONS,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self._local = threading.local()

    def _client(self, *, force_new: bool = False):
        client = None if force_new else getattr(self._local, "client", None)
        if client is None:
            client = boto3.Session().client(
                "bedrock-runtime",
                region_name=self._region,
                config=self._config,
            )
            self._local.client = client
        return client

    def _guarded_call(self, name: str, *args, **kwargs):
        last_exc = None
        for attempt in range(_MAX_RECONNECTS + 1):
            client = self._client(force_new=attempt > 0)
            try:
                return getattr(client, name)(*args, **kwargs)
            except _CONNECTION_ERRORS as exc:
                last_exc = exc
                self._local.client = None
                if attempt < _MAX_RECONNECTS:
                    delay = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** attempt))
                    delay += random.uniform(0, delay / 2)
                    log.warning(
                        "bedrock-runtime.%s connection error (%s); rebuilding client, retry %d/%d in %.1fs",
                        name,
                        type(exc).__name__,
                        attempt + 1,
                        _MAX_RECONNECTS,
                        delay,
                    )
                    time.sleep(delay)
        raise last_exc

    def __getattr__(self, name):
        if name in _GUARDED_METHODS:
            def _call(*args, **kwargs):
                return self._guarded_call(name, *args, **kwargs)

            return _call
        return getattr(self._client(), name)


def create_bedrock_runtime_client(region: str) -> ThreadLocalBedrockRuntimeClient:
    return ThreadLocalBedrockRuntimeClient(region)
