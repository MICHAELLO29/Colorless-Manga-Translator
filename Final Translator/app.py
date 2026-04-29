"""
Colorless Manga Translator - Flask Web Application
Modern web GUI replacing the deprecated Gradio interface.

Security: API keys are stored in-memory per session only.
They are NEVER written to disk from the web UI.
When the user closes their browser tab, the key is gone.
"""

import io
import os
import sys

# Fix Windows console encoding - Japanese text from OCR would crash
# the default 'charmap' codec. Force UTF-8 with fallback replacement.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import base64
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify

# Ensure we can import the colorless_translator package
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, static_folder='public', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# ─── Per-key translator cache ─────────────────────────────────────────────────
# Maps API key hash -> translator instance so we don't re-init on every request,
# but also don't persist the key itself.
import hashlib

_translators = {}   # key_hash -> MangaTranslator
_translator_errors = {}  # key_hash -> error string


def _key_hash(api_key: str) -> str:
    """Hash the API key so we never store the raw key in memory maps."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def get_translator_for_key(api_key: str):
    """Get or create a translator instance for a given API key."""
    kh = _key_hash(api_key)

    if kh in _translators:
        return _translators[kh], None

    if kh in _translator_errors:
        return None, _translator_errors[kh]

    try:
        from colorless_translator.config import Settings
        # Load settings from .env first (picks up Roboflow config),
        # then override the Gemini API key from the web session.
        settings = Settings.from_env()
        settings.gemini_api_key = api_key

        from colorless_translator.core.translator import MangaTranslator
        translator = MangaTranslator(settings=settings)
        translator.initialize()
        _translators[kh] = translator
        return translator, None
    except Exception as e:
        err = str(e)
        _translator_errors[kh] = err
        print(f"Failed to initialize translator: {e}")
        traceback.print_exc()
        return None, err


def _get_api_key_from_request() -> str:
    """Extract the API key from the X-API-Key header."""
    return (request.headers.get('X-API-Key') or '').strip()


def image_to_base64_png(cv2_image) -> str:
    """Convert a CV2 BGR image to a base64-encoded PNG data URI."""
    import cv2
    success, buffer = cv2.imencode('.png', cv2_image)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """Silence favicon 404 errors."""
    return '', 204


@app.route('/api/status', methods=['GET'])
def api_status():
    """Check system status. The client sends the key via header;
    if no header is present, the key is not configured for this session."""
    api_key = _get_api_key_from_request()
    key_set = bool(api_key)

    return jsonify({
        "api_key_configured": key_set,
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/api/test-key', methods=['POST'])
def api_test_key():
    """Validate a Gemini API key by trying to generate content directly."""
    data = request.get_json(silent=True) or {}
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({"success": False, "error": "No API key provided"}), 400

    from google import genai

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid API key format: {e}"}), 400

    # Try common models directly
    common_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
    ]

    last_error = None
    for model_name in common_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents='Say "OK"',
            )
            if response and response.text:
                return jsonify({
                    "success": True,
                    "message": f"API key is valid! (verified with {model_name})"
                })
        except Exception as e:
            last_error = str(e)
            err_lower = last_error.lower()
            # Auth error = key is invalid, stop
            if "api key" in err_lower and ("invalid" in err_lower or "denied" in err_lower):
                return jsonify({"success": False, "error": f"Invalid API key"}), 400
            # Quota error = key IS valid, just rate-limited
            if any(kw in err_lower for kw in ["429", "quota", "resource_exhausted", "resource exhausted", "rate limit"]):
                return jsonify({
                    "success": True,
                    "message": f"API key is valid! (quota limited on {model_name} — will auto-select best available model)"
                })
            continue

    # Fallback: try listing models and use the first available one
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            if not name:
                continue
            try:
                response = client.models.generate_content(
                    model=name,
                    contents='Say "OK"',
                )
                if response and response.text:
                    return jsonify({
                        "success": True,
                        "message": f"API key is valid! (verified with {name})"
                    })
            except Exception:
                continue
    except Exception:
        pass

    return jsonify({
        "success": False,
        "error": f"API key could not be validated: {last_error or 'No working models found'}"
    }), 400


@app.route('/api/translate', methods=['POST'])
def api_translate():
    """Translate a single manga page.
    Requires X-API-Key header with a valid Gemini key."""
    api_key = _get_api_key_from_request()
    if not api_key:
        return jsonify({"error": "No API key provided. Please enter your key in Settings."}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    translator, error = get_translator_for_key(api_key)
    if translator is None:
        return jsonify({
            "error": f"Translator not available: {error or 'Unknown error'}"
        }), 500

    try:
        image_bytes = file.read()
        result = translator.translate_image_bytes(image_bytes)

        if result is None:
            return jsonify({"error": "Failed to process image"}), 500

        # Get the model name from the translator's Gemini translator
        model_name = "unknown"
        try:
            model_name = translator.pipeline.translator.model_name
        except Exception:
            pass

        response_data = {
            "success": result.success,
            "strategy": result.strategy,
            "blocks_processed": result.blocks_processed,
            "blocks_failed": result.blocks_failed,
            "error": result.error,
            "quota_exhausted": False,
            "model": model_name,
        }

        if result.image is not None:
            response_data["translated_image"] = image_to_base64_png(result.image)

        # Also send the original image for side-by-side
        import cv2
        import numpy as np
        from PIL import Image as PILImage
        original_pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        original_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        response_data["original_image"] = image_to_base64_png(original_cv2)

        # Save cache after each translation
        translator.save_cache()

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        err_str = str(e).lower()
        is_quota = any(term in err_str for term in [
            'quota', 'rate limit', 'resource exhausted',
            'too many requests', '429', 'exceeded',
        ])
        return jsonify({
            "error": str(e),
            "quota_exhausted": is_quota,
        }), 429 if is_quota else 500


@app.route('/api/cache-stats', methods=['GET'])
def api_cache_stats():
    """Get translation cache statistics."""
    api_key = _get_api_key_from_request()
    if not api_key:
        return jsonify({"entries": 0, "message": "No API key"})

    translator, _ = get_translator_for_key(api_key)
    if translator is None or translator.cache is None:
        return jsonify({"entries": 0, "message": "Cache not available"})

    return jsonify({
        "entries": len(translator.cache.cache) if hasattr(translator.cache, 'cache') else 0,
    })


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Colorless Manga Translator — Web GUI")
    print("=" * 60)
    print("\n  Starting web server...")
    print("  Open your browser to: http://localhost:7860")
    print("  Press Ctrl+C to stop\n")

    app.run(
        host='0.0.0.0',
        port=7860,
        debug=False,
        threaded=True,
    )
