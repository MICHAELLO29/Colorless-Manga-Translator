# Colorless Manga Translator — Full Revision Plan

Comprehensive overhaul to fix broken functionality, migrate to the modern Gemini SDK, and replace the non-functional Gradio GUI with a polished, standalone web UI.

---

## User Review Required

> [!CAUTION]
> **Your current API key in `.env` is exposed in the repository and may be revoked/expired.** You will need to generate a **fresh API key** from [Google AI Studio](https://aistudio.google.com/apikey) before running the updated project. I will update the `.env` format and code to use it correctly.

> [!IMPORTANT]
> **Major SDK Change**: The `google-generativeai` package is **deprecated** as of Dec 2025. This plan migrates to the new official `google-genai` SDK. The API surface changes from `genai.GenerativeModel(model).generate_content()` to `genai.Client().models.generate_content(model=...)`.

> [!WARNING]
> **The `web_gui.py` (Gradio UI) is completely broken.** It imports from a non-existent `Colorless_translate.py` file and modules (`translation_quality`, `series_memory`, `advanced_typography`) that don't exist at those import paths. It cannot run at all. This plan replaces it with a new standalone Flask web UI.

---

## Problems Identified

### 1. Broken Gemini API (Critical)
- Uses deprecated `google-generativeai` package with model name `gemini-pro` (no longer available — deprecated/404)
- The model recovery fallback list references `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-pro`, `gemini-1.0-pro` — **all are now deprecated**
- Current `.env` API key is likely expired/revoked since it's hardcoded in the repo

### 2. Broken Web GUI (Critical)
- `web_gui.py` imports from `Colorless_translate.py` — **this file does not exist** in the project
- Imports `translation_quality`, `series_memory`, `advanced_typography` from wrong paths
- The GUI literally cannot start — it will `sys.exit(1)` immediately
- Uses Gradio which adds heavyweight ML dependencies

### 3. Disconnected Modules
- `series_memory.py` and `advanced_typography.py` sit in the `colorless_translator/` root but aren't imported by the core pipeline
- The CLI pipeline works independently but the GUI doesn't connect to it

### 4. Requirements Issues
- References `yolov5` package (not used — project uses `ultralytics` YOLOv8)
- Missing `google-genai` (the new SDK)
- `google-generativeai` is deprecated

---

## Proposed Changes

### Component 1: Gemini SDK Migration

Migrate from the deprecated `google-generativeai` to the new `google-genai` SDK, and update the default model to `gemini-2.5-flash` (the current stable, free-tier model).

#### [MODIFY] [gemini.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/colorless_translator/translation/gemini.py)
- Replace `import google.generativeai as genai` → `from google import genai`
- Change API initialization: `genai.configure(api_key=...)` → `self.client = genai.Client(api_key=...)`
- Change model calls: `self.model.generate_content(prompt)` → `self.client.models.generate_content(model=self.model_name, contents=prompt)`
- Update model recovery list to current models: `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`
- Default model: `gemini-2.5-flash` (stable, free tier, fast)

#### [MODIFY] [settings.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/colorless_translator/config/settings.py)
- Change default `gemini_model_name` from `"gemini-pro"` to `"gemini-2.5-flash"`

#### [MODIFY] [test_gemini_api.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/colorless_translator/test_gemini_api.py)
- Update to use new `google-genai` SDK

#### [MODIFY] [series_memory.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/colorless_translator/series_memory.py)
- Update `ContextAwareTranslator` to use the new SDK client pattern instead of `model.generate_content()`

---

### Component 2: New Web GUI (Flask + Modern UI)

Replace the broken Gradio `web_gui.py` with a standalone Flask web app that uses the existing `colorless_translator` package directly. No additional ML dependencies.

#### [NEW] [app.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/app.py)
- Flask web server that imports `MangaTranslator` from the existing package
- REST API endpoints:
  - `POST /api/translate` — Upload single image, return translated image + metadata
  - `POST /api/translate-batch` — Upload multiple images for batch processing
  - `GET /api/status` — Check API key validity and model status
  - `GET /api/cache-stats` — Cache statistics
- Serves the static frontend

#### [NEW] [templates/index.html](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/templates/index.html)
- Single-page application UI with:
  - Drag-and-drop image upload (single + batch)
  - Side-by-side original vs. translated preview
  - Real-time progress indicator
  - Translation details panel (detected texts, translations, quality scores)
  - Strategy selector (Auto/Action/Dialogue/Standard)
  - Settings panel (API key input, model selection)
  - Export translations as JSON
- Premium dark-theme design with glassmorphism, smooth animations
- Responsive layout — works on desktop and tablet
- No framework dependencies (vanilla HTML/CSS/JS)

#### [DELETE] [web_gui.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/web_gui.py)
- Remove the broken Gradio GUI

---

### Component 3: Requirements & Dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/requirements.txt)
- Remove `google-generativeai` → Add `google-genai`
- Remove `yolov5` (not used, project uses `ultralytics`)
- Add `flask>=3.0.0` for the new web GUI
- Clean up unnecessary dependencies (`gitpython`, `thop`, `seaborn`)

#### [MODIFY] [.env](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/.env)
- Update to placeholder format: `GEMINI_API_KEY=your_api_key_here`
- User will need to get a fresh key from [Google AI Studio](https://aistudio.google.com/apikey)

---

### Component 4: CLI & Package Fixes

#### [MODIFY] [__init__.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/colorless_translator/translation/__init__.py)
- Verify imports are correct after SDK migration

#### [MODIFY] [translator.py](file:///c:/Users/Miggy/Documents/Colorless-Manga-Translator/Final%20Translator/colorless_translator/core/translator.py)
- Add a `translate_image_bytes()` method for the web GUI to use directly with uploaded file bytes instead of file paths

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SDK | `google-genai` | Official recommended SDK; `google-generativeai` is deprecated |
| Model | `gemini-2.5-flash` | Current stable model, free tier, fast, great for text translation |
| Web Framework | Flask | Lightweight, no ML dependencies, perfect for serving a REST API |
| Frontend | Vanilla HTML/CSS/JS | No build step, premium dark-mode design, zero framework overhead |
| Gradio | Remove | Broken imports, large dependency footprint, can't start |

---

## Open Questions

> [!IMPORTANT]
> **You need a new API key.** Please go to [Google AI Studio](https://aistudio.google.com/apikey) and generate a fresh free API key. Do you already have one, or should I proceed with a placeholder and you'll add it later?

---

## Verification Plan

### Automated Tests
1. Run `python -m colorless_translator.test_gemini_api` to verify the new SDK connects and lists models
2. Start the Flask web app with `python app.py` and verify it launches on `http://localhost:7860`
3. Test the web UI by uploading a manga page through the browser and confirming translation completes

### Manual Verification
- Browser test: Upload an image via drag-and-drop, verify translated result appears
- Verify strategy selection, translation details, and export work
- Verify CLI still works: `python translate.py`
