# PyPTO_learning

PyPTO All Docs In One — a collection of PyPTO ecosystem sub-repositories for learning and code review.

## Sub-Repositories

| Path | Repository | Description |
|---|---|---|
| `pypto/` | [hw-native-sys/pypto](https://github.com/hw-native-sys/pypto) | A high-performance programming framework for tile-centric computing |
| `pto-as/` | [zhangstevenunity/PTOAS](https://github.com/zhangstevenunity/PTOAS) | PTO Assembler – compile PTO IR/assembly to bytecode |
| `pto-isa/` | [PTO-ISA/pto-isa](https://github.com/PTO-ISA/pto-isa) | PTO Instruction Set Architecture |

## Getting Started

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/ariusewy/PyPTO_learning.git
```

Or if you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Building Documentation

Install dependencies and build:

```bash
pip install -r requirements-docs.txt
python scripts/gather_docs.py
mkdocs build
```

Serve locally:

```bash
mkdocs serve
```

## Automated Updates

Sub-repositories are automatically updated daily via the
[Update Submodules](.github/workflows/update-submodules.yml) workflow.

Documentation is automatically built and deployed to GitHub Pages on every
push to `main` via the [Build and Deploy Docs](.github/workflows/build-docs.yml) workflow.
