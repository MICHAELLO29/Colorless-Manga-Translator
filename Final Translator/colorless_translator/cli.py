"""Command-line interface for the manga translator."""

import argparse
import sys
from pathlib import Path

from colorless_translator.core.translator import MangaTranslator
from colorless_translator.core.exceptions import ConfigurationError, QuotaExhaustedError
from colorless_translator.config import get_settings


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="colorless_translator",
        description="Professional Manga Translator - Translate Japanese manga to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m colorless_translator                    # Use default folders from .env
  python -m colorless_translator -i ./manga -o ./out  # Specify folders
  python -m colorless_translator --image page1.png    # Translate single image
        """,
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input folder containing manga pages (overrides .env)",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output folder for translated images (overrides .env)",
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Translate a single image file",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output path for single image (use with --image)",
    )
    
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version information",
    )
    
    return parser


def main(args: list = None):
    """Main entry point for CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    if parsed.version:
        from colorless_translator import __version__
        print(f"Colorless Manga Translator v{__version__}")
        return 0
    
    translator = None
    
    try:
        translator = MangaTranslator()
        
        if parsed.image:
            input_path = parsed.image
            output_path = parsed.output_file
            
            if not output_path:
                stem = Path(input_path).stem
                output_path = f"{stem}_translated.png"
            
            success = translator.translate_image(input_path, output_path)
            return 0 if success else 1
        else:
            stats = translator.translate_folder(
                input_folder=parsed.input,
                output_folder=parsed.output,
            )
            return 0 if stats["failed"] == 0 else 1
            
    except ConfigurationError as e:
        print(f"\nConfiguration error: {e}")
        print("Make sure you have a valid .env file with your GEMINI_API_KEY")
        return 1
    except KeyboardInterrupt:
        print(f"\n\nProcess interrupted by user. Progress has been saved.")
        if translator:
            translator.save_cache()
        return 130
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if translator:
            translator.save_cache()
        return 1


if __name__ == "__main__":
    sys.exit(main())
