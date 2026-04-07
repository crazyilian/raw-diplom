from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path

from yadisk_sync_common import (
    DiskClient,
    MAX_WORKERS,
    ProgressLog,
    RemoteEntry,
    download_text,
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

SCAN_STATE = threading.local()
COLLECT_SUBDIRS_THREADS_THRESHOLD = 8


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


def collect_subdirs(
    items: list[tuple[DiskClient, str, Path]],
) -> list[tuple[list[DownloadTask], int]]:
    if len(items) <= COLLECT_SUBDIRS_THREADS_THRESHOLD or getattr(SCAN_STATE, "active", False):
        return [collect_tree(*item) for item in items]

    pool = ThreadPool(min(MAX_WORKERS, len(items)))
    try:
        results = list(pool.imap(collect_tree_in_pool, items))
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return results


def collect_tree_in_pool(args: tuple[DiskClient, str, Path]) -> tuple[list[DownloadTask], int]:
    SCAN_STATE.active = True
    return collect_tree(*args)


def collect_tree(client: DiskClient, remote_dir: str, local_dir: Path) -> tuple[list[DownloadTask], int]:
    remote_entries = client.list_dir(remote_dir)
    plots = split_plots(remote_dir, remote_entries)
    tasks: list[DownloadTask] = []
    total = 1 if plots is not None else 0
    subdirs: list[tuple[DiskClient, str, Path]] = []

    print("COLLECTING   ", remote_dir)

    for name, remote_meta in sorted(remote_entries.items()):
        if name == "plots.zip":
            continue
        remote_path = remote_join(remote_dir, name)
        local_path = local_dir / name

        if remote_meta.type == "dir":
            subdirs.append((client, remote_path, local_path))
        elif remote_meta.type == "file":
            total += 1
            if plots is None:
                tasks.append(DownloadTask(remote_path, local_path, meta=remote_meta))
        else:
            raise RuntimeError(f"Unsupported remote type: {remote_meta.type} at {remote_path}")

    for child_tasks, child_total in collect_subdirs(subdirs):
        total += child_total
        if plots is None:
            tasks.extend(child_tasks)

    if plots is not None:
        return [DownloadTask(remote_dir, local_dir, total, True)], total
    return tasks, total


def collect_tasks(client: DiskClient, remote_dir: str, local_dir: Path) -> list[DownloadTask]:
    res = collect_tree(client, remote_dir, local_dir)[0]
    res.sort(key=lambda task: str(task.local))
    return res


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
