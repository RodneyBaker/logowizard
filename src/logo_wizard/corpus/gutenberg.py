"""Helpers for Project Gutenberg plain-text files."""

import re

_START = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_END = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)


def strip_boilerplate(text: str) -> str:
    """Return only the body of a Gutenberg text, without the license header and footer."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    start = _START.search(text)
    end = _END.search(text)
    begin = start.end() if start else 0
    finish = end.start() if end else len(text)
    return text[begin:finish].strip("\n")
