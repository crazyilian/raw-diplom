from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import requests

BASE_URL = "https://cloud-api.yandex.net/v1/disk"
API_TIMEOUT = 5
FILE_TIMEOUT = 10
LIST_LIMIT = 1000
MAX_WORKERS = 8
RETRY_ATTEMPTS = 5
RETRY_DELAY = 2
TMP_DIR = Path(__file__).resolve().parent / "tmp"
WORKDIR = Path.cwd().resolve()

T = TypeVar("T")
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class RemoteEntry:
    type: str
    md5: str | None


def log_line(text: str) -> None:
    with PRINT_LOCK:
        print(text)


def local_text(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(WORKDIR))
    except ValueError:
        return str(resolved)


def upload_text(local_path: Path, remote_path: str) -> str:
    return f"{local_text(local_path)} -> {remote_path}"


def download_text(remote_path: str, local_path: Path) -> str:
    return f"{remote_path} -> {local_text(local_path)}"


def _response_payload(response: requests.Response) -> object:
    try:
        return response.json()
    except Exception:
        return response.text


def _raise_for_status(response: requests.Response, action: str, *, allow_statuses: tuple[int, ...] = ()) -> None:
    if response.status_code < 400 or response.status_code in allow_statuses:
        return
    raise RuntimeError(f"{action} failed: {response.status_code} {_response_payload(response)}")


def retry(action: str, fn: Callable[[], T]) -> T:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except (requests.RequestException, RuntimeError) as error:
            if attempt == RETRY_ATTEMPTS:
                raise
            log_line(f"WARN   {action} failed ({error}), retry {attempt + 1}/{RETRY_ATTEMPTS}")
            time.sleep(RETRY_DELAY)
    raise AssertionError("unreachable")


class DiskClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Authorization": f"OAuth {self.token}"})
            self._local.session = session
        return session

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        allow_statuses: tuple[int, ...] = (),
    ) -> requests.Response:
        response = self._session().request(method, BASE_URL + path, params=params, timeout=API_TIMEOUT)
        try:
            _raise_for_status(response, f"{method} {path}", allow_statuses=allow_statuses)
            return response
        except Exception:
            response.close()
            raise

    def get_meta(self, remote_path: str, *, allow_missing: bool = False) -> dict | None:
        response = self._request(
            "GET",
            "/resources",
            params={"path": remote_path},
            allow_statuses=(404,) if allow_missing else (),
        )
        try:
            if response.status_code == 404:
                return None
            return response.json()
        finally:
            response.close()

    def ensure_dir(self, remote_path: str) -> bool:
        meta = self.get_meta(remote_path, allow_missing=True)
        if meta is None:
            response = self._request("PUT", "/resources", params={"path": remote_path}, allow_statuses=(409,))
            response.close()
            return False
        if meta.get("type") != "dir":
            raise RuntimeError(f"Remote file blocks folder: {remote_path}")
        return True

    def list_dir(self, remote_dir: str) -> dict[str, RemoteEntry]:
        items: dict[str, RemoteEntry] = {}
        offset = 0

        while True:
            response = self._request(
                "GET",
                "/resources",
                params={"path": remote_dir, "limit": LIST_LIMIT, "offset": offset},
            )
            try:
                data = response.json()
            finally:
                response.close()

            if data.get("type") != "dir":
                raise RuntimeError(f"Remote path is not a folder: {remote_dir}")

            embedded = data.get("_embedded") or {}
            batch = embedded.get("items") or []
            total = embedded.get("total", 0)

            for item in batch:
                name = logical_remote_name(item["name"], item["type"])
                items[name] = RemoteEntry(type=item["type"], md5=_lower_or_none(item.get("md5")))

            offset += len(batch)
            if offset >= total or not batch:
                return items

    def file_md5(self, remote_path: str, hint: RemoteEntry | None = None) -> str | None:
        if hint is not None and hint.type == "file" and hint.md5 is not None:
            return hint.md5

        meta = self.get_meta(remote_file_path(remote_path), allow_missing=True)
        if meta is None:
            return None
        if meta.get("type") != "file":
            raise RuntimeError(f"Remote path is not a file: {remote_path}")
        return _lower_or_none(meta.get("md5"))

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        action = f"upload {upload_text(local_path, remote_path)}"

        def operation() -> None:
            response = self._request(
                "GET",
                "/resources/upload",
                params={"path": remote_file_path(remote_path), "overwrite": "true"},
            )
            try:
                href = response.json()["href"]
            finally:
                response.close()

            with local_path.open("rb") as fh:
                upload = requests.put(href, data=fh, timeout=FILE_TIMEOUT)
                try:
                    _raise_for_status(upload, action)
                finally:
                    upload.close()

        retry(action, operation)

    def download_file(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        action = f"download {download_text(remote_path, local_path)}"

        def operation() -> None:
            response = self._request(
                "GET",
                "/resources/download",
                params={"path": remote_file_path(remote_path)},
            )
            try:
                href = response.json()["href"]
            finally:
                response.close()

            try:
                with requests.get(href, stream=True, timeout=FILE_TIMEOUT) as download:
                    _raise_for_status(download, action)
                    with local_path.open("wb") as fh:
                        for chunk in download.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                fh.write(chunk)
            except (requests.RequestException, RuntimeError):
                local_path.unlink(missing_ok=True)
                raise

        retry(action, operation)


class ProgressLog:
    def __init__(self, total: int) -> None:
        self.total = total
        self.current = 0
        self._lock = threading.Lock()

    def line(self, action: str, text: str) -> None:
        with self._lock:
            self.current += 1
            log_line(f"[{self.current}/{self.total}] {action:<6} {text}")


def run_in_pool(items: Iterable[T], fn: Callable[[T], None]) -> None:
    items = list(items)
    if not items:
        return

    pool = ThreadPool(min(MAX_WORKERS, len(items)))
    try:
        for _ in pool.imap(fn, items):
            pass
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        raise


def _lower_or_none(value: object) -> str | None:
    return str(value).lower() if value else None


def normalize_remote_path(remote_path: str) -> str:
    remote_path = remote_path.strip()
    if remote_path == "disk:":
        return "disk:/"
    if remote_path.startswith("disk:/"):
        return remote_path.rstrip("/") or "disk:/"
    return f"disk:/{remote_path.lstrip('/')}".rstrip("/") or "disk:/"


def remote_join(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name


def remote_file_path(remote_path: str) -> str:
    return remote_path + "fast" if remote_path.endswith(".zip") else remote_path


def logical_remote_name(name: str, resource_type: str) -> str:
    if resource_type == "file" and name.endswith(".zipfast"):
        return name[:-4]
    return name


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_local_dir(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def remove_local_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def temp_zip_path(prefix: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".zip", dir=TMP_DIR)
    os.close(fd)
    return Path(name)


def make_temp_zip(source_dir: Path) -> Path:
    zip_path = temp_zip_path(f"{source_dir.name}_")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(source_dir):
            dirs.sort()
            files.sort()
            root_path = Path(root)
            rel_root = root_path.relative_to(source_dir)

            if rel_root != Path(".") and not dirs and not files:
                zf.writestr(rel_root.as_posix() + "/", b"")

            for name in files:
                path = root_path / name
                zf.write(path, path.relative_to(source_dir).as_posix())
    return zip_path


def extract_zip_to_clean_dir(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path inside zip: {member.filename}")
        remove_local_path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(target_dir)
