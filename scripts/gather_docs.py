#!/usr/bin/env python3
"""Gather documentation from submodules into the docs/ directory for MkDocs."""

import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


def copy_tree(src: str, dst: str) -> None:
    """Copy a directory tree, creating dst if needed."""
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_tree(s, d)
        else:
            shutil.copy2(s, d)


def main() -> None:
    # --- pypto ---
    pypto_src = os.path.join(REPO_ROOT, "pypto", "docs", "en")
    pypto_dst = os.path.join(DOCS_DIR, "pypto")
    if os.path.isdir(pypto_src):
        shutil.rmtree(pypto_dst, ignore_errors=True)
        copy_tree(pypto_src, pypto_dst)
        # Add a convenience getting-started page at top level
        gs_src = os.path.join(pypto_src, "user", "00-getting_started.md")
        gs_dst = os.path.join(pypto_dst, "getting-started.md")
        if os.path.isfile(gs_src):
            shutil.copy2(gs_src, gs_dst)
        print(f"Copied pypto docs from {pypto_src}")
    else:
        print(f"WARNING: pypto docs not found at {pypto_src}")

    # --- pto-as ---
    ptoas_src = os.path.join(REPO_ROOT, "pto-as", "docs")
    ptoas_dst = os.path.join(DOCS_DIR, "pto-as")
    readme_src = os.path.join(REPO_ROOT, "pto-as", "README.md")
    if os.path.isdir(ptoas_src):
        shutil.rmtree(ptoas_dst, ignore_errors=True)
        copy_tree(ptoas_src, ptoas_dst)
        if os.path.isfile(readme_src):
            shutil.copy2(readme_src, os.path.join(ptoas_dst, "README.md"))
        print(f"Copied pto-as docs from {ptoas_src}")
    else:
        print(f"WARNING: pto-as docs not found at {ptoas_src}")

    # --- pto-isa ---
    ptoisa_src = os.path.join(REPO_ROOT, "pto-isa", "docs")
    ptoisa_dst = os.path.join(DOCS_DIR, "pto-isa")
    if os.path.isdir(ptoisa_src):
        shutil.rmtree(ptoisa_dst, ignore_errors=True)
        copy_tree(ptoisa_src, ptoisa_dst)
        print(f"Copied pto-isa docs from {ptoisa_src}")
    else:
        print(f"WARNING: pto-isa docs not found at {ptoisa_src}")

    print("Done gathering docs.")


if __name__ == "__main__":
    main()
