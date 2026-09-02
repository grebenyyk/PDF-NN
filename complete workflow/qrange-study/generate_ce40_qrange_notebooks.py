#!/usr/bin/env python3
"""Generate and optionally execute Ce40-based descriptor notebooks."""

from generate_ceo2_qrange_notebooks import main


if __name__ == "__main__":
    raise SystemExit(main(default_dataset="ce40"))