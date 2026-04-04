from __future__ import annotations

import argparse
from pathlib import Path

from yadisk_sync_common import (
    DiskClient,
    ensure_local_dir,
    extract_zip_to_clean_dir,
    file_md5,
    normalize_remote_path,
    remote_join,
    remove_local_path,
    temp_zip_name,
)


def sync_download_file(client: DiskClient, remote_file: str, remote_meta: dict, local_file: Path, overwrite: bool) -> None:
    remote_md5 = client.remote_md5(remote_file, remote_meta)

    if local_file.exists() and local_file.is_file() and remote_md5 is not None and file_md5(local_file) == remote_md5:
        print(f"OK     {local_file}")
        return

    if local_file.exists() and local_file.is_dir():
        if not overwrite:
            print(f"SKIP   {local_file} (folder blocks file download, use --overwrite-different-size)")
            return
        remove_local_path(local_file)

    if local_file.exists() and local_file.is_file() and not overwrite:
        print(f"SKIP   {local_file} (different md5, use --overwrite-different-size)")
        return

    action = "REGET" if local_file.exists() else "GET"
    print(f"{action:<6} {remote_file} -> {local_file}")
    client.download_file(remote_file, local_file)


def refresh_plots_from_zip(client: DiskClient, remote_zip_path: str, local_plots_dir: Path) -> None:
    if local_plots_dir.exists():
        remove_local_path(local_plots_dir)

    tmp_zip = temp_zip_name(remote_zip_path, "download")
    try:
        print(f"GET    {remote_zip_path} -> {local_plots_dir}")
        client.download_file(remote_zip_path, tmp_zip)
        extract_zip_to_clean_dir(tmp_zip, local_plots_dir)
    finally:
        tmp_zip.unlink(missing_ok=True)


def sync_download_dir(client: DiskClient, remote_dir: str, local_dir: Path, overwrite: bool) -> None:
    ensure_local_dir(local_dir, overwrite=overwrite)
    remote_entries = client.list_children(remote_dir)

    if "plots.zip" in remote_entries and remote_entries["plots.zip"]["type"] != "file":
        raise RuntimeError(f"Ambiguous remote folder: plots.zip exists but is not a file in {remote_dir}")
    if "plots.zip" in remote_entries and "plots" in remote_entries:
        raise RuntimeError(f"Ambiguous remote folder: both plots.zip and plots/ exist in {remote_dir}")

    for name in sorted(remote_entries):
        remote_meta = remote_entries[name]
        remote_path = remote_join(remote_dir, name)

        if name == "plots.zip" and remote_meta["type"] == "file":
            refresh_plots_from_zip(client, remote_path, local_dir / "plots")
            continue

        local_path = local_dir / name

        if remote_meta["type"] == "dir":
            sync_download_dir(client, remote_path, local_path, overwrite)
        elif remote_meta["type"] == "file":
            sync_download_file(client, remote_path, remote_meta, local_path, overwrite)
        else:
            print(f"SKIP   {remote_path} (unsupported remote type: {remote_meta['type']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Yandex Disk folder into a local folder with special handling for plots.zip -> plots/")
    parser.add_argument("--local", required=True, help="Local folder")
    parser.add_argument("--remote-root", required=True, help="Remote folder on Yandex Disk")
    parser.add_argument("--token", required=True, help="Yandex OAuth token")
    parser.add_argument("--overwrite-different-size", action="store_true", help="Overwrite local regular files when the md5 is different")
    args = parser.parse_args()

    local_root = Path(args.local).expanduser().resolve()
    if local_root.name == "plots":
        raise RuntimeError("--local must point to the parent folder, not directly to plots/")

    client = DiskClient(args.token)
    remote_root = normalize_remote_path(args.remote_root)
    sync_download_dir(client, remote_root, local_root, args.overwrite_different_size)


if __name__ == "__main__":
    main()
