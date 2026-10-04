#!/usr/bin/env python3
"""
ERCOT Public API client shared by the SCED and COP downloaders.

download_ercot_sced_waveforms.py and download_ercot_cop_snapshots.py use
``ErcotClient`` for authentication, throttling and retries.

ERCOT Public API documentation:
  https://developer.ercot.com/applications/pubapi/
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.ercot.com/api/public-reports"
ENDPOINT = "/np3-965-er/60_load_res_data_in_sced"
AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
SCOPE = f"openid {CLIENT_ID} offline_access"
# ERCOT limit is 30 requests/min; >=2 seconds/request stays under the limit.
DEFAULT_SLEEP_SECONDS = 2.1


def request_json(url: str, *, headers=None, data=None, method=None, timeout=90, binary=False):
    req = urllib.request.Request(
        url,
        headers=headers or {},
        data=data,
        method=method or ("POST" if data is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return raw if binary else json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}\n{body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def authenticate(username: str, password: str) -> str:
    payload = urllib.parse.urlencode(
        {
            "username": username,
            "password": password,
            "grant_type": "password",
            "scope": SCOPE,
            "client_id": CLIENT_ID,
            "response_type": "id_token",
        }
    ).encode("utf-8")
    response = request_json(
        AUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=payload,
        method="POST",
        timeout=90,
    )
    # ERCOT documentation calls for the ID token in the Authorization header.
    token = response.get("id_token") or response.get("access_token")
    if not token:
        msg = response.get("error_description") or response.get("error") or str(response)
        raise RuntimeError(f"ERCOT authentication failed: {msg}")
    return token


class ErcotClient:
    def __init__(self, username: str, password: str, subscription_key: str, sleep_seconds: float):
        self.username = username
        self.password = password
        self.subscription_key = subscription_key
        self.sleep_seconds = sleep_seconds
        self.token = None
        self.last_request_at = 0.0

    def refresh_token(self):
        print("Authenticating with ERCOT Public API...")
        self.token = authenticate(self.username, self.password)

    def _throttle(self):
        elapsed = time.monotonic() - self.last_request_at
        wait = self.sleep_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def get(self, params: dict, retries: int = 5, *, endpoint=ENDPOINT, binary=False):
        # Archive URLs originate in ERCOT metadata. Never send credentials to
        # an arbitrary host or an unexpected API path.
        url = endpoint if endpoint.startswith("https://") else API_BASE + endpoint
        if not url.startswith(API_BASE + "/"):
            raise ValueError("Download URL must stay within ERCOT public-reports API")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        if not self.token:
            self.refresh_token()
        for attempt in range(1, retries + 1):
            self._throttle()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Ocp-Apim-Subscription-Key": self.subscription_key,
                # ERCOT's public-reports gateway negotiates archive downloads
                # with the same JSON media type as metadata requests. The body
                # is still ZIP bytes when ?download=<docId> is present.
                "Accept": "application/json",
                "User-Agent": "University-of-Tsukuba-ADER-Research/1.0",
            }
            try:
                result = request_json(url, headers=headers, timeout=180, binary=binary)
                self.last_request_at = time.monotonic()
                return result
            except RuntimeError as exc:
                msg = str(exc)
                self.last_request_at = time.monotonic()
                if "HTTP 401" in msg and attempt < retries:
                    print("Token rejected/expired; obtaining a new token...")
                    self.refresh_token()
                    continue
                if "HTTP 429" in msg and attempt < retries:
                    backoff = max(10, 5 * attempt)
                    print(f"Rate limited; waiting {backoff}s...")
                    time.sleep(backoff)
                    continue
                if any(code in msg for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")) and attempt < retries:
                    backoff = 5 * attempt
                    print(f"ERCOT server error; retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                raise
        raise RuntimeError("Request failed after retries")
