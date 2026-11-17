"""Stage and push the reproduction repository.

The report zip stays as it is; this is the repo a reader clones when they want to run
things. It contains every script, every result file the notebook reads, the annotation
exports, and the small `subset_ids.json`. It does not contain the images or the 49 MB
ground-truth dump: `scripts/fetch_subset.py --ids ...` re-downloads exactly the committed
sample.

Usage: python scripts/make_repo.py [--push]
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE = os.path.join(ROOT, "repo")
REPO = "cs5495-mmhs-xai"

DIRS = ["scripts", "src", "figures", "annotate", "templates"]
FILES = [
    "REPO_README.md",
    "requirements.txt",
    "Course Project-2026A.ipynb",
    "Course Project-2026A.pdf",
    "Course Project-2026A.html",
    "findings.md",
    "TEAM_SUMMARY.md",
    "CONTRIBUTION.md",
]
LEGACY = ("r0", "baseline_")


def run(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-800:])
        print(result.stderr[-800:])
        raise SystemExit(f"failed: {' '.join(args)}")
    return result.stdout.strip()


def clean(path: str) -> None:
    """rmtree that survives git's read-only object files on Windows."""
    def onexc(func, target, error):
        os.chmod(target, stat.S_IWRITE)
        func(target)
    shutil.rmtree(path, onexc=onexc)


def main() -> None:
    if os.path.exists(STAGE):
        clean(STAGE)
    os.makedirs(STAGE)

    for name in DIRS:
        source = os.path.join(ROOT, name)
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(STAGE, name),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))

    results = os.path.join(STAGE, "refine-logs")
    os.makedirs(results, exist_ok=True)
    for name in os.listdir(os.path.join(ROOT, "refine-logs")):
        if name.startswith(LEGACY):
            continue
        source = os.path.join(ROOT, "refine-logs", name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(results, name))

    for name in FILES:
        source = os.path.join(ROOT, name)
        if not os.path.exists(source):
            print("missing, skipped:", name)
            continue
        target = "README.md" if name == "REPO_README.md" else name
        shutil.copy2(source, os.path.join(STAGE, target))

    data_dir = os.path.join(STAGE, "data", "MMHS150K")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "data", "MMHS150K", "subset_ids.json"),
                 os.path.join(data_dir, "subset_ids.json"))

    with open(os.path.join(STAGE, ".gitignore"), "w", encoding="utf-8") as handle:
        handle.write("data/MMHS150K/img_resized/\ndata/MMHS150K/MMHS150K_GT.json\n"
                     "data/MMHS150K/_cache/\nmodels/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n"
                     "cp_stage.html\n")

    run("git", "init", "-b", "main", cwd=STAGE)
    run("git", "add", "-A", cwd=STAGE)
    run("git", "-c", "user.name=Botherain",
        "-c", "user.email=Botherain@users.noreply.github.com",
        "commit", "-m", "CS5495 XAI audit of a multimodal hate-speech detector: code, results, report",
        cwd=STAGE)

    files = run("git", "ls-files", cwd=STAGE).splitlines()
    size = sum(os.path.getsize(os.path.join(STAGE, name)) for name in files) / 1e6
    print(f"staged {len(files)} files, {size:.1f} MB")

    if "--push" in sys.argv:
        created = subprocess.run(
            ["gh", "repo", "create", REPO, "--public", "--source", ".", "--push",
             "--description", "Reproduction package for a CS5495 XAI audit of a multimodal hate-speech detector"],
            cwd=STAGE, capture_output=True, text=True)
        print(created.stdout.strip() or created.stderr.strip()[-300:])
    print("stage:", STAGE)


if __name__ == "__main__":
    main()
