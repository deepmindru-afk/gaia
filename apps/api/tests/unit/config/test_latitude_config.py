"""Unit tests for the Latitude ingest-endpoint resolution.

Settings wins over process env, always: the SDK freezes EXPORTER_URL at
import time, so this is the line that decides self-hosted vs Latitude Cloud.
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

from latitude_telemetry.env import env as latitude_env

from app.config.latitude import resolve_exporter_endpoint


def _settings(url: str) -> SimpleNamespace:
    return SimpleNamespace(
        LATITUDE_API_KEY="key-1",
        LATITUDE_PROJECT="gaia",
        LATITUDE_TELEMETRY_URL=url,
    )


class TestResolveExporterEndpoint:
    def test_settings_url_wins_over_process_env(self) -> None:
        previous = os.environ.get("LATITUDE_TELEMETRY_URL")
        try:
            with patch("app.config.latitude.settings", _settings("http://selfhost:3002")):
                os.environ["LATITUDE_TELEMETRY_URL"] = "https://stale.example.com"

                assert resolve_exporter_endpoint() == "http://selfhost:3002"
                assert os.environ["LATITUDE_TELEMETRY_URL"] == "http://selfhost:3002"
        finally:
            if previous is None:
                os.environ.pop("LATITUDE_TELEMETRY_URL", None)
            else:
                os.environ["LATITUDE_TELEMETRY_URL"] = previous

    def test_rebinds_frozen_sdk_value(self) -> None:
        previous_url = latitude_env.EXPORTER_URL
        previous_env = os.environ.get("LATITUDE_TELEMETRY_URL")
        try:
            with patch("app.config.latitude.settings", _settings("http://selfhost:3002")):
                resolve_exporter_endpoint()

                assert latitude_env.EXPORTER_URL == "http://selfhost:3002"
        finally:
            latitude_env.EXPORTER_URL = previous_url
            if previous_env is None:
                os.environ.pop("LATITUDE_TELEMETRY_URL", None)
            else:
                os.environ["LATITUDE_TELEMETRY_URL"] = previous_env
