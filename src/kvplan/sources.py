"""Read model files from the Hugging Face Hub or a local directory.

Only small files (config.json, index files) are read whole. Weight files are read
with HTTP range requests, so no weights are downloaded.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://huggingface.co"
USER_AGENT = "kvplan (https://github.com/ShawnNotFound/kvplan)"


class SourceError(Exception):
    """A model's files could not be read."""


class GatedRepoError(SourceError):
    """The repository needs an access token (gated or private)."""


def find_token() -> str | None:
    """Return a Hugging Face token from the environment or the huggingface_hub token file."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_file = hf_home / "token"
    if token_file.is_file():
        token = token_file.read_text().strip()
        return token or None
    return None


class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never send the token to another host (e.g. the CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_DropAuthOnCrossHostRedirect)


class Source:
    """Interface for reading a model's files."""

    name: str

    def list_files(self) -> list[str]:
        raise NotImplementedError

    def read(self, filename: str) -> bytes | None:
        """Read a whole file, or None if it does not exist."""
        raise NotImplementedError

    def read_range(self, filename: str, start: int, end: int) -> tuple[bytes, int | None]:
        """Read bytes [start, end] inclusive. Returns the bytes and the total file size if known."""
        raise NotImplementedError

    def read_json(self, filename: str) -> dict | None:
        data = self.read(filename)
        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise SourceError(f"{self.name}: {filename} is not valid JSON ({e})") from e


class LocalSource(Source):
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.name = str(self.path)

    def list_files(self) -> list[str]:
        if self.path.is_file():
            return [self.path.name]
        return sorted(str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file())

    def _file(self, filename: str) -> Path:
        return self.path if self.path.is_file() and filename == self.path.name else self.path / filename

    def read(self, filename: str) -> bytes | None:
        f = self._file(filename)
        return f.read_bytes() if f.is_file() else None

    def read_range(self, filename: str, start: int, end: int) -> tuple[bytes, int | None]:
        f = self._file(filename)
        if not f.is_file():
            raise SourceError(f"{self.name}: {filename} not found")
        with f.open("rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1), f.stat().st_size


class HubSource(Source):
    def __init__(self, repo_id: str, revision: str = "main", token: str | None = None,
                 endpoint: str = DEFAULT_ENDPOINT, timeout: float = 60.0):
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.name = repo_id

    def _open(self, url: str, rng: tuple[int, int] | None = None):
        headers = {"User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if rng is not None:
            headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
        try:
            return _opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                hint = "set HF_TOKEN or run `huggingface-cli login`" if not self.token else \
                    "your token does not have access; accept the model's licence on its Hugging Face page"
                raise GatedRepoError(f"{self.repo_id} is gated or private ({e.code}): {hint}") from e
            if e.code == 404:
                return None
            raise SourceError(f"{self.repo_id}: HTTP {e.code} for {url}") from e
        except urllib.error.URLError as e:
            raise SourceError(f"{self.repo_id}: cannot reach {self.endpoint} ({e.reason})") from e

    def _file_url(self, filename: str) -> str:
        return (f"{self.endpoint}/{self.repo_id}/resolve/{urllib.parse.quote(self.revision, safe='')}/"
                f"{urllib.parse.quote(filename)}")

    def list_files(self) -> list[str]:
        url = f"{self.endpoint}/api/models/{self.repo_id}/revision/{urllib.parse.quote(self.revision, safe='')}"
        resp = self._open(url)
        if resp is None:
            raise SourceError(f"{self.repo_id}: repository or revision '{self.revision}' not found")
        with resp:
            info = json.load(resp)
        return sorted(s["rfilename"] for s in info.get("siblings", []))

    def read(self, filename: str) -> bytes | None:
        resp = self._open(self._file_url(filename))
        if resp is None:
            return None
        with resp:
            return resp.read()

    def read_range(self, filename: str, start: int, end: int) -> tuple[bytes, int | None]:
        resp = self._open(self._file_url(filename), (start, end))
        if resp is None:
            raise SourceError(f"{self.repo_id}: {filename} not found")
        with resp:
            body = resp.read()
            total = None
            m = re.match(r"bytes \d+-\d+/(\d+)", resp.headers.get("Content-Range", ""))
            if m:
                total = int(m.group(1))
            elif resp.status == 200:
                # Server ignored the range and sent the whole file.
                total = len(body)
                body = body[start:end + 1]
        return body, total


def open_source(model: str, revision: str = "main", token: str | None = None) -> Source:
    """A local path if it exists, otherwise a Hugging Face repo id."""
    if os.path.exists(model):
        return LocalSource(model)
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", model):
        raise SourceError(f"'{model}' is neither a local path nor a Hugging Face repo id (owner/name)")
    return HubSource(model, revision=revision, token=token or find_token())
