#!/usr/bin/env python3
"""Patch a verified portable-reader CSS edge case caused by classic scrollbars.

The bundled reader makes its sticky top bar ``100vw`` wide and recent Chromium
counts the vertical scrollbar inside ``vw`` but outside ``clientWidth``.  That
creates an otherwise empty 8 px horizontal overflow.  The report content is
still produced by the official portable builder; this deterministic patch only
changes that one top-bar rule before the official verifier is run again.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    text = args.html.read_text(encoding="utf-8")
    replacements = {
        "width:100vw": "width:100%",
        "margin-right:calc(50% - 50vw)": "margin-right:0",
        "margin-left:calc(50% - 50vw)": "margin-left:0",
    }
    for old, new in replacements.items():
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"expected exactly one {old!r} rule, found {count}")
        text = text.replace(old, new)
    marker = "data-sitian-scrollbar-fix"
    if marker in text:
        raise RuntimeError("portable scrollbar fix is already present")
    override = (
        f'<style {marker}="true">'
        ".analytics-top-bar{width:100%!important;margin-left:0!important;"
        "margin-right:0!important;left:0!important;right:auto!important;"
        "transform:none!important;max-width:100%!important}"
        "</style>"
    )
    if "</head>" not in text:
        raise RuntimeError("portable HTML has no </head> marker")
    text = text.replace("</head>", override + "</head>", 1)
    temporary = args.html.with_suffix(args.html.suffix + ".layout-tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, args.html)
    print(f"patched portable top-bar scrollbar overflow: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
