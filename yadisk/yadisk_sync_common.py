from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional, TypeVar

import requests

BASE_URL = "https://cloud-api.yandex.net/v1/disk"
API_TIMEOUT = (10, 60)
FILE_TIMEOUT = (10, 300)
LIST_LIMIT = 1000
TMP_DIR = Path("./tmp")
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}

T = TypeVar("T")


class DiskTransientError(RuntimeError):
    pass


def _response_payload(response: requests.Response) -> object:
    try:
        return response.json()
    except Exception:
        return response.text


def _raise_for_status(response: requests.Response, action: str, *, allow_statuses: tuple[int, ...] = ()) -> requests.Response:
    if response.status_code < 400 or response.status_code in allow_statuses:
        return response

    message = f"{action} failed: {response.status_code} {_response_payload(response)}"
    if response.status_code in RETRY_STATUSES:
        raise DiskTransientError(message)
    raise RuntimeError(message)


def _retry(action: str, operation: Callable[[], T]) -> T:
    last_error: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except (requests.RequestException, DiskTransientError) as error:
            last_error = error
            if attempt == RETRY_ATTEMPTS:
                break
            print(f"WARN   {action} ({error}), retry {attempt + 1}/{RETRY_ATTEMPTS}")
            time.sleep(RETRY_DELAY)

    raise DiskTransientError(f"{action} failed after {RETRY_ATTEMPTS} attempts: {last_error}")


class DiskClient:
    def __init__(self, token: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {token}"})

    def _request(self, method: str, url: str, *, params=None, allow_statuses: tuple[int, ...] = ()) -> requests.Response:
        action = f"{method} {url}"
        def operation() -> requests.Response:
            response = self.session.request(method, url, params=params, timeout=API_TIMEOUT)
            try:
                return _raise_for_status(response, action, allow_statuses=allow_statuses)
            except Exception:
                response.close()
                raise

        return _retry(action, operation)

    def get_meta(self, remote_path: str) -> Optional[dict]:
        response = self._request(
            "GET",
            f"{BASE_URL}/resources",
            params={"path": remote_path},
            allow_statuses=(404,),
        )
        if response.status_code == 404:
            return None
        return response.json()

    def get_file_meta(self, remote_path: str) -> Optional[dict]:
        response = self._request(
            "GET",
            f"{BASE_URL}/resources",
            params={"path": remote_file_path(remote_path)},
            allow_statuses=(404,),
        )
        if response.status_code == 404:
            return None
        return response.json()

    def ensure_dir(self, remote_path: str) -> bool:
        meta = self.get_meta(remote_path)
        if meta is None:
            self._request(
                "PUT",
                f"{BASE_URL}/resources",
                params={"path": remote_path},
                allow_statuses=(409,),
            )
            return False
        if meta.get("type") != "dir":
            raise RuntimeError(f"Remote path is not a folder: {remote_path}")
        return True

    def list_children(self, remote_dir: str) -> Dict[str, dict]:
        meta = self.get_meta(remote_dir)
        if meta is None:
            raise RuntimeError(f"Remote folder does not exist: {remote_dir}")
        if meta.get("type") != "dir":
            raise RuntimeError(f"Remote path is not a folder: {remote_dir}")

        items: Dict[str, dict] = {}
        offset = 0

        while True:
            response = self._request(
                "GET",
                f"{BASE_URL}/resources",
                params={"path": remote_dir, "limit": LIST_LIMIT, "offset": offset},
            )
            data = response.json()
            embedded = data.get("_embedded") or {}
            batch = embedded.get("items") or []
            total = embedded.get("total", 0)

            for item in batch:
                logical_name = logical_remote_name(item["name"], item["type"])
                items[logical_name] = {
                    "name": logical_name,
                    "type": item["type"],
                    "size": item.get("size"),
                    "md5": item.get("md5"),
                }

            offset += len(batch)
            if offset >= total or not batch:
                break

        return items

    def upload_file(self, local_path: Path, remote_path: str, *, overwrite: bool) -> None:
        response = self._request(
            "GET",
            f"{BASE_URL}/resources/upload",
            params={"path": remote_file_path(remote_path), "overwrite": str(overwrite).lower()},
        )
        href = response.json()["href"]

        _retry(
            f"upload {local_path} -> {remote_path}",
            lambda: self._upload_once(href, local_path),
        )

    def download_file(self, remote_path: str, local_path: Path) -> None:
        response = self._request(
            "GET",
            f"{BASE_URL}/resources/download",
            params={"path": remote_file_path(remote_path)},
        )
        href = response.json()["href"]

        local_path.parent.mkdir(parents=True, exist_ok=True)
        _retry(
            f"download {remote_path} -> {local_path}",
            lambda: self._download_once(href, local_path),
        )

    def remote_md5(self, remote_path: str, remote_meta: dict | None = None) -> Optional[str]:
        if remote_meta is not None and remote_meta.get("type") == "file" and remote_meta.get("md5"):
            return str(remote_meta["md5"]).lower()

        meta = self.get_file_meta(remote_path)
        if meta is None:
            return None
        if meta.get("type") != "file":
            raise RuntimeError(f"Remote path is not a file: {remote_path}")

        md5 = meta.get("md5")
        return str(md5).lower() if md5 else None

    def _upload_once(self, href: str, local_path: Path) -> None:
        with local_path.open("rb") as fh:
            response = requests.put(href, data=fh, timeout=FILE_TIMEOUT)
            try:
                _raise_for_status(response, "Upload")
            finally:
                response.close()

    def _download_once(self, href: str, local_path: Path) -> None:
        try:
            with requests.get(href, stream=True, timeout=FILE_TIMEOUT) as download_response:
                _raise_for_status(download_response, "Download")
                with local_path.open("wb") as fh:
                    for chunk in download_response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
        except (requests.RequestException, DiskTransientError):
            local_path.unlink(missing_ok=True)
            raise


def normalize_remote_path(remote_path: str) -> str:
    remote_path = remote_path.strip()
    if remote_path == "disk:":
        return "disk:/"
    if remote_path.startswith("disk:/"):
        return remote_path.rstrip("/") or "disk:/"
    return f"disk:/{remote_path.lstrip('/')}".rstrip("/") or "disk:/"


def remote_join(base: str, name: str) -> str:
    if base.endswith("/"):
        return base + name
    return base + "/" + name


def remote_file_path(remote_path: str) -> str:
    if remote_path.endswith(".zip"):
        return remote_path + "fast"
    return remote_path


def logical_remote_name(name: str, resource_type: str) -> str:
    if resource_type == "file" and name.endswith(".zipfast"):
        return name[:-4]
    return name


def ensure_local_dir(path: Path, *, overwrite: bool) -> bool:
    if path.exists() and path.is_file():
        if not overwrite:
            raise RuntimeError(f"Local file blocks folder: {path}")
        path.unlink()
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return existed


def remove_local_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def temp_zip_name(key: str, prefix: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return TMP_DIR / f"{prefix}_{digest}.zip"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_temp_zip(source_dir: Path) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{source_dir.name}_", suffix=".zip", dir=TMP_DIR)
    os.close(fd)
    zip_path = Path(tmp_name)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(source_dir):
            dirs.sort()
            files.sort()
            root_path = Path(root)
            rel_root = root_path.relative_to(source_dir)

            if rel_root != Path(".") and not dirs and not files:
                zf.writestr(rel_root.as_posix() + "/", b"")

            for filename in files:
                abs_path = root_path / filename
                rel_path = abs_path.relative_to(source_dir)
                zf.write(abs_path, rel_path.as_posix())

    return zip_path


def extract_zip_to_clean_dir(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path inside zip: {member.filename}")

        if target_dir.exists():
            remove_local_path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(target_dir)
