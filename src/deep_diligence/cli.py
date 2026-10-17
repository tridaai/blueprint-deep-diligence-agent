"""Command line entry point: `deep-diligence demo` runs the offline demo."""

from __future__ import annotations

import sys

from . import demo


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo.main()
        return
    print("usage: deep-diligence demo    # or: make demo / make serve / make evals")


if __name__ == "__main__":
    main()
