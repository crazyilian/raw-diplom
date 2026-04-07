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
class DownloadTask:
    remote: str
    local: Path
    count: int = 1
    is_dir: bool = False
    meta: RemoteEntry | None = None


def split_plots(remote_dir: str, remote_entries: dict[str, RemoteEntry]) -> RemoteEntry | None:
    plots = remote_entries.get("plots.zip")
    if plots is not None and plots.type != "file":
        raise RuntimeError(f"Remote path is not a file: {remote_join(remote_dir, 'plots.zip')}")
    if plots is not None and "plots" in remote_entries:
        raise RuntimeError(f"Ambiguous remote folder: both plots.zip and plots/ exist in {remote_dir}")
    return plots


def count_lines(client: DiskClient, remote_dir: str, remote_entries: dict[str, RemoteEntry] | None = None) -> int:
    remote_entries = remote_entries or client.list_dir(remote_dir)
    total = 1 if split_plots(remote_dir, remote_entries) is not None else 0

    for name, remote_meta in sorted(remote_entries.items()):
        if name == "plots.zip":
            continue
        remote_path = remote_join(remote_dir, name)
        if remote_meta.type == "dir":
            total += count_lines(client, remote_path)
        elif remote_meta.type == "file":
            total += 1
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")
    return total


def collect_tasks(client: DiskClient, remote_dir: str, local_dir: Path) -> list[DownloadTask]:
    remote_entries = client.list_dir(remote_dir)
    if split_plots(remote_dir, remote_entries) is not None:
        return [DownloadTask(remote_dir, local_dir, count_lines(client, remote_dir, remote_entries), True)]

    ensure_local_dir(local_dir)
    tasks: list[DownloadTask] = []
    for name, remote_meta in sorted(remote_entries.items()):
        remote_path = remote_join(remote_dir, name)
        local_path = local_dir / name

        if remote_meta.type == "dir":
            tasks.extend(collect_tasks(client, remote_path, local_path))
        elif remote_meta.type == "file":
            tasks.append(DownloadTask(remote_path, local_path, meta=remote_meta))
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")
    return tasks


def sync_file(
    client: DiskClient,
    remote_path: str,
    remote_meta: RemoteEntry,
    local_path: Path,
    progress: ProgressLog,
) -> bool:
    if local_path.is_file() and client.file_md5(remote_path, remote_meta) == file_md5(local_path):
        progress.line("OK", local_text(local_path))
        return False

    existed = local_path.exists()
    if local_path.is_dir():
        remove_local_path(local_path)

    client.download_file(remote_path, local_path)
    progress.line("REGET" if existed else "GET", download_text(remote_path, local_path))
    return True


def refresh_plots(client: DiskClient, remote_path: str, local_path: Path, progress: ProgressLog) -> None:
    if local_path.exists():
        remove_local_path(local_path)

    zip_path = temp_zip_path("download_")
    try:
        client.download_file(remote_path, zip_path)
        extract_zip_to_clean_dir(zip_path, local_path)
        progress.line("GET", download_text(remote_path, local_path))
    finally:
        zip_path.unlink(missing_ok=True)


def sync_tree(client: DiskClient, remote_dir: str, local_dir: Path, progress: ProgressLog) -> bool:
    ensure_local_dir(local_dir)
    remote_entries = client.list_dir(remote_dir)
    plots = split_plots(remote_dir, remote_entries)
    changed = False

    for name, remote_meta in sorted(remote_entries.items()):
        if name == "plots.zip":
            continue
        remote_path = remote_join(remote_dir, name)
        local_path = local_dir / name

        if remote_meta.type == "dir":
            changed = sync_tree(client, remote_path, local_path, progress) or changed
        elif remote_meta.type == "file":
            changed = sync_file(client, remote_path, remote_meta, local_path, progress) or changed
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")

    if plots is None:
        return changed

    local_plots = local_dir / "plots"
    if not changed and local_plots.is_dir():
        progress.line("SKIP", local_text(local_plots))
        return False

    refresh_plots(client, remote_join(remote_dir, "plots.zip"), local_plots, progress)
    return True


def run_task(client: DiskClient, progress: ProgressLog, task: DownloadTask) -> None:
    if task.is_dir:
        sync_tree(client, task.remote, task.local, progress)
        return
    if task.meta is None:
        raise RuntimeError(f"Missing remote meta: {task.remote}")
    sync_file(client, task.remote, task.meta, task.local, progress)


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
    tasks = collect_tasks(client, remote_root, local_root)
    if not tasks:
        return

    log_line(f"INFO   start download tasks: {len(tasks)}")
    run_in_pool(tasks, partial(run_task, client, ProgressLog(sum(task.count for task in tasks))))


if __name__ == "__main__":
    main()
