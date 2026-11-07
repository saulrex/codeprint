#!/usr/bin/env python3
"""Входная точка harvest (модуль 8, см. docs/ARCHITECTURE.md).

Из этого файла собирается standalone exe:
    pyinstaller --onefile harvest.py
"""
from codeprint.harvest import main

if __name__ == "__main__":
    raise SystemExit(main())
