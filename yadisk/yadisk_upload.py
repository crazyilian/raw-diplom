from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from yadisk_sync_common import (
    DiskClient,
    file_md5,
    make_temp_zip,
    normalize_remote_path,
    remote_join,
)


def scan_local_entries(local_dir: Path) -> Dict[str, Path]:
    entries = {child.name: child for child in local_dir.iterdir()}
    if "plots" in entries and "plots.zip" in entries and entries["plots"].is_dir() and entries["plots.zip"].is_file():
        raise RuntimeError(f"Ambiguous local folder: both plots/ and plots.zip exist in {local_dir}")
    return entries


def sync_upload_file(client: DiskClient, local_file: Path, remote_file: str, remote_meta: dict | None, overwrite: bool) -> bool:
    if remote_meta is None:
        print(f"UPLOAD {local_file} -> {remote_file}")
        client.upload_file(local_file, remote_file, overwrite=True)
        return False

    if remote_meta["type"] != "file":
        raise RuntimeError(f"Remote folder blocks file upload: {remote_file}")

    local_md5 = file_md5(local_file)
    remote_md5 = client.remote_md5(remote_file, remote_meta)

    if remote_md5 is not None and remote_md5 == local_md5:
        print(f"OK     {local_file}")
        return True

    if not overwrite:
        print(f"SKIP   {local_file} (different md5, use --overwrite-different-size)")
        return False

    print(f"REPUT  {local_file} -> {remote_file}")
    client.upload_file(local_file, remote_file, overwrite=True)
    return False


def sync_upload_plots(client: DiskClient, local_plots_dir: Path, remote_zip_path: str, remote_meta: dict | None, overwrite: bool) -> bool:
    tmp_zip = make_temp_zip(local_plots_dir)
    try:
        zip_md5 = file_md5(tmp_zip)

        if remote_meta is not None and remote_meta["type"] != "file":
            raise RuntimeError(f"Remote folder blocks plots.zip upload: {remote_zip_path}")

        remote_md5 = client.remote_md5(remote_zip_path, remote_meta) if remote_meta is not None else None
        if remote_md5 is not None and remote_md5 == zip_md5:
            print(f"OK     {local_plots_dir} (plots.zip md5 matches)")
            return True

        if remote_meta is not None and not overwrite:
            print(f"SKIP   {local_plots_dir} (plots.zip has different md5, use --overwrite-different-size)")
            return False

        action = "REPUT" if remote_meta is not None else "UPLOAD"
        print(f"{action:<6} {local_plots_dir} -> {remote_zip_path}")
        client.upload_file(tmp_zip, remote_zip_path, overwrite=True)
        return False
    finally:
        tmp_zip.unlink(missing_ok=True)


def sync_upload_dir(client: DiskClient, local_dir: Path, remote_dir: str, overwrite: bool) -> bool:
    remote_dir_existed = client.ensure_dir(remote_dir)
    local_entries = scan_local_entries(local_dir)
    remote_entries = client.list_children(remote_dir)

    if "plots" in local_entries and local_entries["plots"].is_dir() and "plots.zip" in remote_entries and remote_entries["plots.zip"]["type"] != "file":
        raise RuntimeError(f"Ambiguous remote folder: plots.zip exists but is not a file in {remote_dir}")

    others_exact = remote_dir_existed

    for name in sorted(local_entries):
        local_path = local_entries[name]
        if local_path.is_dir() and name == "plots":
            continue

        remote_meta = remote_entries.get(name)
        remote_path = remote_join(remote_dir, name)

        if local_path.is_dir():
            if remote_meta is not None and remote_meta["type"] != "dir":
                raise RuntimeError(f"Remote file blocks folder upload: {remote_path}")
            child_exact = sync_upload_dir(client, local_path, remote_path, overwrite)
        elif local_path.is_file():
            child_exact = sync_upload_file(client, local_path, remote_path, remote_meta, overwrite)
        else:
            print(f"SKIP   {local_path} (not a regular file or folder)")
            child_exact = False

        others_exact = others_exact and child_exact

    represented_remote_names = set(local_entries)
    if "plots" in local_entries and local_entries["plots"].is_dir():
        represented_remote_names.discard("plots")
        represented_remote_names.add("plots.zip")

    for remote_name in remote_entries:
        if remote_name not in represented_remote_names:
            others_exact = False

    plots_exact = True
    local_plots = local_entries.get("plots")
    remote_plots_meta = remote_entries.get("plots.zip")

    if local_plots is not None and local_plots.is_dir():
        remote_zip_path = remote_join(remote_dir, "plots.zip")

        if others_exact and remote_plots_meta is not None:
            print(f"OK     {local_plots} (fast path, plots.zip assumed synced)")
            plots_exact = True
        else:
            plots_exact = sync_upload_plots(client, local_plots, remote_zip_path, remote_plots_meta, overwrite)

    return others_exact and plots_exact


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local folder to Yandex Disk with special handling for plots/ -> plots.zip")
    parser.add_argument("--local", required=True, help="Local folder")
    parser.add_argument("--remote-root", required=True, help="Remote folder on Yandex Disk")
    parser.add_argument("--token", required=True, help="Yandex OAuth token")
    parser.add_argument("--overwrite-different-md5", action="store_true", help="Overwrite remote files when the md5 is different")
    args = parser.parse_args()

    local_root = Path(args.local).expanduser().resolve()
    if not local_root.exists() or not local_root.is_dir():
        raise RuntimeError(f"Local folder does not exist: {local_root}")
    if local_root.name == "plots":
        raise RuntimeError("--local must point to the parent folder, not directly to plots/")

    client = DiskClient(args.token)
    remote_root = normalize_remote_path(args.remote_root)
    sync_upload_dir(client, local_root, remote_root, args.overwrite_different_md5)


if __name__ == "__main__":
    main()
