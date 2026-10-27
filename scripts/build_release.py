"""Thin standalone project entrypoint."""

import sys

from gstar_opd_formal.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "build-release")
    main()
