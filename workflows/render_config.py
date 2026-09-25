#!/usr/bin/env python3
"""Render one workflow YAML template from exported environment variables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from string import Template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rendered = Template(args.template.read_text()).substitute(os.environ)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
