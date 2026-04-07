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
    remote_file_path,
    remote_join,
    run_in_pool,
    upload_text,
)


@dataclass(frozen=True)
class UploadFileTask:
    local_path: Path
    remote_path: str
    count: int = 1


@dataclass(frozen=True)
class UploadObjectTask:
    local_dir: Path
    remote_dir: str
    count: int


UploadTask = UploadFileTask | UploadObjectTask


def iter_children(path: Path) -> list[Path]:
    return sorted(path.iterdir(), key=lambda child: child.name)


def has_direct_plots(local_dir: Path) -> bool:
    return (local_dir / "plots").is_dir()


def count_upload_lines(local_dir: Path) -> int:
    total = 0
    for child in iter_children(local_dir):
        if child.name == "plots" and child.is_dir():
            total += 1
        elif child.is_dir():
            total += count_upload_lines(child)
        elif child.is_file():
            total += 1
        else:
            raise RuntimeError(f"Unsupported local path: {child}")
    return total


def prepare_upload_dirs(client: DiskClient, local_dir: Path, remote_dir: str) -> None:
    if has_direct_plots(local_dir):
        return
    client.ensure_dir(remote_dir)
    for child in iter_children(local_dir):
        if child.is_dir():
            prepare_upload_dirs(client, child, remote_join(remote_dir, child.name))


def collect_upload_tasks(local_dir: Path, remote_dir: str) -> list[UploadTask]:
    if has_direct_plots(local_dir):
        return [UploadObjectTask(local_dir=local_dir, remote_dir=remote_dir, count=count_upload_lines(local_dir))]

    tasks: list[UploadTask] = []
    for child in iter_children(local_dir):
        remote_path = remote_join(remote_dir, child.name)
        if child.is_dir():
            tasks.extend(collect_upload_tasks(child, remote_path))
        elif child.is_file():
            tasks.append(UploadFileTask(local_path=child, remote_path=remote_path))
        else:
            raise RuntimeError(f"Unsupported local path: {child}")
    return tasks


def sync_upload_file(
    client: DiskClient,
    local_file: Path,
    remote_file: str,
    remote_meta: RemoteEntry | None,
    progress: ProgressLog,
) -> bool:
    if remote_meta is None:
        client.upload_file(local_file, remote_file)
        progress.line("PUT", upload_text(local_file, remote_file))
        return False

    if remote_meta.type != "file":
        raise RuntimeError(f"Remote folder blocks file upload: {remote_file}")

    remote_md5 = client.file_md5(remote_file, remote_meta)
    if remote_md5 is not None and file_md5(local_file) == remote_md5:
        progress.line("OK", local_text(local_file))
        return True

    client.upload_file(local_file, remote_file)
    progress.line("REPUT", upload_text(local_file, remote_file))
    return False


def sync_upload_plots(
    client: DiskClient,
    local_plots_dir: Path,
    remote_zip_path: str,
    remote_meta: RemoteEntry | None,
    progress: ProgressLog,
) -> bool:
    if remote_meta is not None and remote_meta.type != "file":
        raise RuntimeError(f"Remote folder blocks plots upload: {remote_zip_path}")

    zip_path = make_temp_zip(local_plots_dir)
    try:
        remote_md5 = client.file_md5(remote_zip_path, remote_meta) if remote_meta is not None else None
        if remote_md5 is not None and file_md5(zip_path) == remote_md5:
            progress.line("OK", local_text(local_plots_dir))
            return True

        client.upload_file(zip_path, remote_zip_path)
        progress.line("REPUT" if remote_meta is not None else "PUT", upload_text(local_plots_dir, remote_zip_path))
        return False
    finally:
        zip_path.unlink(missing_ok=True)


def sync_upload_tree(client: DiskClient, local_dir: Path, remote_dir: str, progress: ProgressLog) -> bool:
    remote_existed = client.ensure_dir(remote_dir)
    remote_entries = client.list_dir(remote_dir)
    others_exact = remote_existed
    represented: set[str] = set()

    for child in iter_children(local_dir):
        if child.name == "plots" and child.is_dir():
            represented.add("plots.zip")
            continue

        represented.add(child.name)
        remote_path = remote_join(remote_dir, child.name)
        remote_meta = remote_entries.get(child.name)

        if child.is_dir():
            if remote_meta is not None and remote_meta.type != "dir":
                raise RuntimeError(f"Remote file blocks folder upload: {remote_path}")
            others_exact = sync_upload_tree(client, child, remote_path, progress) and others_exact
        elif child.is_file():
            others_exact = sync_upload_file(client, child, remote_path, remote_meta, progress) and others_exact
        else:
            raise RuntimeError(f"Unsupported local path: {child}")

    for remote_name in remote_entries:
        if remote_name not in represented:
            others_exact = False

    local_plots = local_dir / "plots"
    remote_plots = remote_entries.get("plots.zip")
    if not local_plots.is_dir():
        return others_exact

    remote_plots_path = remote_join(remote_dir, "plots.zip")
    if others_exact and remote_plots is not None:
        progress.line("SKIP", local_text(local_plots))
        return True

    sync_upload_plots(client, local_plots, remote_plots_path, remote_plots, progress)
    return False


def run_upload_file_task(client: DiskClient, task: UploadFileTask, progress: ProgressLog) -> None:
    meta = client.get_meta(remote_file_path(task.remote_path), allow_missing=True)
    if meta is None:
        remote_meta = None
    else:
        remote_meta = RemoteEntry(
            type=str(meta["type"]),
            md5=str(meta["md5"]).lower() if meta.get("md5") else None,
        )
    sync_upload_file(client, task.local_path, task.remote_path, remote_meta, progress)


def run_upload_task(client: DiskClient, progress: ProgressLog, task: UploadTask) -> None:
    if isinstance(task, UploadFileTask):
        run_upload_file_task(client, task, progress)
        return
    sync_upload_tree(client, task.local_dir, task.remote_dir, progress)


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

    log_line("INFO   prepare remote folders")
    prepare_upload_dirs(client, local_root, remote_root)
    log_line("INFO   scan local tree")
    tasks = collect_upload_tasks(local_root, remote_root)
    if not tasks:
        return

    progress = ProgressLog(sum(task.count for task in tasks))
    log_line(f"INFO   start upload tasks: {len(tasks)}")
    run_in_pool(tasks, partial(run_upload_task, client, progress))


if __name__ == "__main__":
    main()
