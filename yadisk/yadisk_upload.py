from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from yadisk_sync_common import (
    DiskClient,
    ProgressLog,
    RemoteEntry,
    file_md5,
    local_text,
    log_line,
    make_temp_zip,
    normalize_remote_path,
    remote_entry,
    remote_file_path,
    remote_join,
    remote_parent,
    run_in_pool,
    upload_text,
)


@dataclass(frozen=True)
class UploadTask:
    local: Path
    remote: str
    count: int = 1
    is_dir: bool = False


def children(path: Path) -> list[Path]:
    return sorted(path.iterdir(), key=lambda child: child.name)


def count_lines(local_dir: Path) -> int:
    total = 0
    for child in children(local_dir):
        if child.is_file():
            total += 1
        elif child.is_dir():
            total += 1 if child.name == "plots" else count_lines(child)
        else:
            raise RuntimeError(f"Unsupported local path: {child}")
    return total


def collect_tasks(local_dir: Path, remote_dir: str) -> list[UploadTask]:
    if (local_dir / "plots").is_dir():
        return [UploadTask(local_dir, remote_dir, count_lines(local_dir), True)]

    tasks: list[UploadTask] = []
    for child in children(local_dir):
        remote_path = remote_join(remote_dir, child.name)
        if child.is_dir():
            tasks.extend(collect_tasks(child, remote_path))
        elif child.is_file():
            tasks.append(UploadTask(child, remote_path))
        else:
            raise RuntimeError(f"Unsupported local path: {child}")
    return tasks


def upload_if_needed(
    client: DiskClient,
    source_path: Path,
    shown_path: Path,
    remote_path: str,
    remote_meta: RemoteEntry | None,
    progress: ProgressLog,
) -> bool:
    if remote_meta is not None and remote_meta.type != "file":
        raise RuntimeError(f"Remote folder blocks file upload: {remote_path}")

    if remote_meta is not None and client.file_md5(remote_path, remote_meta) == file_md5(source_path):
        progress.line("OK", local_text(shown_path))
        return True

    if remote_meta is None:
        client.ensure_dir(remote_parent(remote_path))
    client.upload_file(source_path, remote_path)
    progress.line("REPUT" if remote_meta is not None else "PUT", upload_text(shown_path, remote_path))
    return False


def sync_tree(
    client: DiskClient,
    local_dir: Path,
    remote_dir: str,
    progress: ProgressLog,
    remote_exists: bool | None = None,
) -> bool:
    if remote_exists is None:
        meta = remote_entry(client.get_meta(remote_dir, allow_missing=True))
        if meta is None:
            remote_entries = {}
        elif meta.type == "dir":
            remote_entries = client.list_dir(remote_dir)
        else:
            raise RuntimeError(f"Remote file blocks folder upload: {remote_dir}")
    else:
        remote_entries = client.list_dir(remote_dir) if remote_exists else {}

    exact = True
    names: set[str] = set()

    for child in children(local_dir):
        if child.name == "plots" and child.is_dir():
            names.add("plots.zip")
            continue

        names.add(child.name)
        remote_path = remote_join(remote_dir, child.name)
        remote_meta = remote_entries.get(child.name)

        if child.is_dir():
            if remote_meta is not None and remote_meta.type != "dir":
                raise RuntimeError(f"Remote file blocks folder upload: {remote_path}")
            exact = sync_tree(client, child, remote_path, progress, remote_meta is not None) and exact
        elif child.is_file():
            exact = upload_if_needed(client, child, child, remote_path, remote_meta, progress) and exact
        else:
            raise RuntimeError(f"Unsupported local path: {child}")

    if any(name not in names for name in remote_entries):
        exact = False

    plots_dir = local_dir / "plots"
    if not plots_dir.is_dir():
        return exact

    remote_path = remote_join(remote_dir, "plots.zip")
    remote_meta = remote_entries.get("plots.zip")
    if exact and remote_meta is not None:
        progress.line("SKIP", local_text(plots_dir))
        return True

    zip_path = make_temp_zip(plots_dir)
    try:
        return upload_if_needed(client, zip_path, plots_dir, remote_path, remote_meta, progress)
    finally:
        zip_path.unlink(missing_ok=True)


def run_task(client: DiskClient, progress: ProgressLog, task: UploadTask) -> None:
    if task.is_dir:
        sync_tree(client, task.local, task.remote, progress)
        return
    upload_if_needed(
        client,
        task.local,
        task.local,
        task.remote,
        remote_entry(client.get_meta(remote_file_path(task.remote), allow_missing=True)),
        progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local folder to Yandex Disk")
    parser.add_argument("--local", required=True, help="Local folder")
    parser.add_argument("--remote-root", required=True, help="Remote folder on Yandex Disk")
    parser.add_argument("--token", required=True, help="Yandex OAuth token")
    args = parser.parse_args()

    local_root = Path(args.local).expanduser().resolve()
    if not local_root.is_dir():
        raise RuntimeError(f"Local folder does not exist: {local_root}")

    client = DiskClient(args.token)
    remote_root = normalize_remote_path(args.remote_root)

    log_line("INFO   scan local tree")
    tasks = collect_tasks(local_root, remote_root)
    if not tasks:
        return

    log_line(f"INFO   start upload tasks: {len(tasks)}")
    run_in_pool(tasks, partial(run_task, client, ProgressLog(sum(task.count for task in tasks))))


if __name__ == "__main__":
    main()
