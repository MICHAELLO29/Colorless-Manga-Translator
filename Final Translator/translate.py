#!/usr/bin/env python3
"""
Colorless Manga Translator - Main Entry Point

This script provides backwards compatibility with the original Colorless_translate.py.
For the new modular interface, use: python -m colorless_translator

Usage:
    python translate.py                    # Translate all images in configured folders
    python translate.py --help             # Show help
"""

from colorless_translator.cli import main

if __name__ == "__main__":
    main()
