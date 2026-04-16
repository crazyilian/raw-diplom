from __future__ import annotations

import argparse
import re
import threading
from collections import Counter
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

from yadisk_sync_common import DiskClient, MAX_WORKERS, normalize_remote_path, remote_join

SCAN_STATE = threading.local()
THREADS_THRESHOLD = 30
RUN_RE = re.compile(r"^(\d{3})-.+")


@dataclass(frozen=True)
class RemoteDir:
    path: str
    name: str
    files: frozenset[str]
    dirs: tuple["RemoteDir", ...]


def scan_subdirs(items: list[tuple[DiskClient, str, str]]) -> list[RemoteDir]:
    if len(items) <= THREADS_THRESHOLD or getattr(SCAN_STATE, "active", False):
        return [scan_tree(*item) for item in items]

    pool = ThreadPool(min(MAX_WORKERS, len(items)))
    try:
        result = list(pool.imap(scan_tree_in_pool, items))
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return result


def scan_tree_in_pool(args: tuple[DiskClient, str, str]) -> RemoteDir:
    SCAN_STATE.active = True
    return scan_tree(*args)


def scan_tree(client: DiskClient, remote_dir: str, name: str) -> RemoteDir:
    if name == "checkpoints" or name.startswith("debug") or name.startswith("bug"):
        entries = {}
    else:
        print("SCAN   ", remote_dir, name)
        entries = client.list_dir(remote_dir)

    files: set[str] = set()
    subdirs: list[tuple[DiskClient, str, str]] = []

    for child_name, meta in sorted(entries.items()):
        child_path = remote_join(remote_dir, child_name)
        if meta.type == "file":
            files.add(child_name)
        elif meta.type == "dir":
            subdirs.append((client, child_path, child_name))
        else:
            raise RuntimeError(f"Unsupported remote type: {meta.type} at {child_path}")

    return RemoteDir(remote_dir, name, frozenset(files), tuple(scan_subdirs(subdirs)))


def is_run(node: RemoteDir) -> bool:
    return "history.json" in node.files


def walk_families(node: RemoteDir) -> list[RemoteDir]:
    families = [node] if any(is_run(child) for child in node.dirs) else []
    for child in node.dirs:
        families.extend(walk_families(child))
    return families


def family_problems(node: RemoteDir) -> list[str]:
    problems: list[str] = []
    counts: Counter[int] = Counter()

    for child in node.dirs:
        if not is_run(child):
            problems.append(f"DIR   {child.path} is not a run")
            continue

        if "summary.json" not in child.files:
            problems.append(f"RUN   {child.path} missing summary.json")

        match = RUN_RE.match(child.name)
        if match is None or int(match.group(1)) < 1:
            problems.append(f"RUN   {child.path} has invalid name")
            continue

        counts[int(match.group(1))] += 1

    for number in range(1, max(counts, default=0) + 1):
        count = counts.get(number, 0)
        if count < 3:
            problems.append(f"MISS  {node.path}/{number:03d}-* missing {3 - count} run(s)")
        elif count > 3:
            problems.append(f"EXTRA {node.path}/{number:03d}-* has {count - 3} extra run(s)")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Check family/run structure on Yandex Disk")
    parser.add_argument("--remote-root", required=True, help="Remote folder on Yandex Disk")
    parser.add_argument("--token", required=True, help="Yandex OAuth token")
    args = parser.parse_args()

    root_path = normalize_remote_path(args.remote_root)
    root_name = root_path.rsplit("/", 1)[-1] or "disk:/"
    root = scan_tree(DiskClient(args.token), root_path, root_name)
    print('\n\n')

    bad = False
    for family in walk_families(root):
        problems = family_problems(family)
        if not problems:
            continue
        bad = True
        print(f"FAMILY {family.path}")
        for line in problems:
            print(line)

    if not bad:
        print("OK")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
