from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from yadisk_sync_common import (
    DiskClient,
    ProgressLog,
    RemoteEntry,
    download_text,
    ensure_local_dir,
    extract_zip_to_clean_dir,
    file_md5,
    local_text,
    log_line,
    normalize_remote_path,
    remote_join,
    remove_local_path,
    run_in_pool,
    temp_zip_path,
)


@dataclass(frozen=True)
class DownloadFileTask:
    remote_path: str
    remote_meta: RemoteEntry
    local_path: Path
    count: int = 1


@dataclass(frozen=True)
class DownloadObjectTask:
    remote_dir: str
    local_dir: Path
    count: int


DownloadTask = DownloadFileTask | DownloadObjectTask


def validate_remote_dir(remote_dir: str, remote_entries: dict[str, RemoteEntry]) -> None:
    plots_meta = remote_entries.get("plots.zip")
    if plots_meta is not None and plots_meta.type != "file":
        raise RuntimeError(f"Remote path is not a file: {remote_join(remote_dir, 'plots.zip')}")
    if plots_meta is not None and "plots" in remote_entries:
        raise RuntimeError(f"Ambiguous remote folder: both plots.zip and plots/ exist in {remote_dir}")


def count_download_lines(
    client: DiskClient,
    remote_dir: str,
    remote_entries: dict[str, RemoteEntry] | None = None,
) -> int:
    remote_entries = remote_entries or client.list_dir(remote_dir)
    validate_remote_dir(remote_dir, remote_entries)

    total = 1 if "plots.zip" in remote_entries else 0
    for name in sorted(remote_entries):
        if name == "plots.zip":
            continue

        remote_meta = remote_entries[name]
        remote_path = remote_join(remote_dir, name)
        if remote_meta.type == "dir":
            total += count_download_lines(client, remote_path)
        elif remote_meta.type == "file":
            total += 1
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")
    return total


def collect_download_tasks(client: DiskClient, remote_dir: str, local_dir: Path) -> list[DownloadTask]:
    remote_entries = client.list_dir(remote_dir)
    validate_remote_dir(remote_dir, remote_entries)

    if "plots.zip" in remote_entries:
        return [
            DownloadObjectTask(
                remote_dir=remote_dir,
                local_dir=local_dir,
                count=count_download_lines(client, remote_dir, remote_entries),
            )
        ]

    ensure_local_dir(local_dir)
    tasks: list[DownloadTask] = []
    for name in sorted(remote_entries):
        remote_meta = remote_entries[name]
        remote_path = remote_join(remote_dir, name)
        local_path = local_dir / name

        if remote_meta.type == "dir":
            tasks.extend(collect_download_tasks(client, remote_path, local_path))
        elif remote_meta.type == "file":
            tasks.append(DownloadFileTask(remote_path=remote_path, remote_meta=remote_meta, local_path=local_path))
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")
    return tasks


def sync_download_file(
    client: DiskClient,
    remote_file: str,
    remote_meta: RemoteEntry,
    local_file: Path,
    progress: ProgressLog,
) -> bool:
    remote_md5 = client.file_md5(remote_file, remote_meta)
    if local_file.is_file() and remote_md5 is not None and file_md5(local_file) == remote_md5:
        progress.line("OK", local_text(local_file))
        return False

    existed = local_file.exists()
    if local_file.is_dir():
        remove_local_path(local_file)

    client.download_file(remote_file, local_file)
    progress.line("REGET" if existed else "GET", download_text(remote_file, local_file))
    return True


def refresh_plots_from_zip(client: DiskClient, remote_zip_path: str, local_plots_dir: Path, progress: ProgressLog) -> None:
    if local_plots_dir.exists():
        remove_local_path(local_plots_dir)

    zip_path = temp_zip_path("download_")
    try:
        client.download_file(remote_zip_path, zip_path)
        extract_zip_to_clean_dir(zip_path, local_plots_dir)
        progress.line("GET", download_text(remote_zip_path, local_plots_dir))
    finally:
        zip_path.unlink(missing_ok=True)


def sync_download_tree(client: DiskClient, remote_dir: str, local_dir: Path, progress: ProgressLog) -> bool:
    ensure_local_dir(local_dir)
    remote_entries = client.list_dir(remote_dir)
    validate_remote_dir(remote_dir, remote_entries)

    plots_meta = remote_entries.pop("plots.zip", None)
    changed = False

    for name in sorted(remote_entries):
        remote_meta = remote_entries[name]
        remote_path = remote_join(remote_dir, name)
        local_path = local_dir / name

        if remote_meta.type == "dir":
            changed = sync_download_tree(client, remote_path, local_path, progress) or changed
        elif remote_meta.type == "file":
            changed = sync_download_file(client, remote_path, remote_meta, local_path, progress) or changed
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")

    if plots_meta is None:
        return changed

    local_plots_dir = local_dir / "plots"
    if not changed and local_plots_dir.is_dir():
        progress.line("SKIP", local_text(local_plots_dir))
        return False

    refresh_plots_from_zip(client, remote_join(remote_dir, "plots.zip"), local_plots_dir, progress)
    return True


def run_download_task(client: DiskClient, progress: ProgressLog, task: DownloadTask) -> None:
    if isinstance(task, DownloadFileTask):
        sync_download_file(client, task.remote_path, task.remote_meta, task.local_path, progress)
        return
    sync_download_tree(client, task.remote_dir, task.local_dir, progress)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Yandex Disk folder into a local folder")
    parser.add_argument("--local", required=True, help="Local folder")
    parser.add_argument("--remote-root", required=True, help="Remote folder on Yandex Disk")
    parser.add_argument("--token", required=True, help="Yandex OAuth token")
    args = parser.parse_args()

    client = DiskClient(args.token)
    remote_root = normalize_remote_path(args.remote_root)
    local_root = Path(args.local).expanduser().resolve()

    log_line("INFO   scan remote tree")
    tasks = collect_download_tasks(client, remote_root, local_root)
    if not tasks:
        return

    progress = ProgressLog(sum(task.count for task in tasks))
    log_line(f"INFO   start download tasks: {len(tasks)}")
    run_in_pool(tasks, partial(run_download_task, client, progress))


if __name__ == "__main__":
    main()
