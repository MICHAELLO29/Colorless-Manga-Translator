# Colorless Manga Translator

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Roboflow](https://img.shields.io/badge/Roboflow-Detection-orange)]()
[![Gemini](https://img.shields.io/badge/Gemini_2.0-Translation-purple)]()

An AI-powered manga translation pipeline that automatically detects Japanese text in speech bubbles, translates it to English using Google Gemini, and renders clean typeset output — all through a modern web interface.

---

## Features

- **Cloud-Based Detection** — Uses a custom-trained [Roboflow](https://roboflow.com/) model for accurate speech bubble and text detection (no local GPU required)
- **AI Translation** — Powered by Google Gemini 2.0 for high-quality Japanese-to-English translation
- **Smart Inpainting** — Automatically erases Japanese text and fills speech bubbles with matched background colour
- **Auto Font Sizing** — Binary-search algorithm finds the optimal font size for each bubble
- **Column Merging** — Detects adjacent vertical text columns and merges them into horizontal English layout
- **Web GUI** — Drag-and-drop interface with side-by-side original/translated comparison
- **Multi-Page Support** — Process multiple manga pages in one session
- **Translation Cache** — Caches translations to reduce API calls on repeated text

---

## Sample Output

### Original Page
![Original Manga Page](Final%20Translator/manga%20pages/1.png)

### Translated Page
![Translated Manga Page](Final%20Translator/output/2.png)

---

## Prerequisites

- **Python 3.11.x** (recommended — other versions may have compatibility issues)
- **Google Gemini API Key** — Free from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Roboflow API Key** — Free from [Roboflow](https://app.roboflow.com/) (1,000 inferences/month on free tier)

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MICHAELLO29/Colorless-Manga-Translator.git
cd Colorless-Manga-Translator/Final\ Translator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will automatically download the manga-ocr model (~400MB). This only happens once.

### 3. Configure API Keys

Create a `.env` file in the `Final Translator/` directory:

```env
# Google Gemini API Key (required for translation)
GEMINI_API_KEY=your_gemini_api_key_here

# Roboflow Cloud Detection (required for text detection)
ROBOFLOW_API_KEY=your_roboflow_api_key_here
USE_ROBOFLOW=true
ROBOFLOW_MODEL_ID=bubble-text-detector-k5qgg/1
```

**How to get your keys:**

| Key | Where to Get It | Free Tier |
|-----|----------------|-----------|
| Gemini API Key | [Google AI Studio](https://aistudio.google.com/app/apikey) | 1,500 requests/day |
| Roboflow API Key | [Roboflow Dashboard](https://app.roboflow.com/) → Settings → API Keys | 1,000 inferences/month |

### 4. Run the Application

```bash
python app.py
```

Open your browser to **http://localhost:7860**

---

## How to Use

1. **Enter your Gemini API Key** in the web interface when prompted
2. **Upload manga pages** using drag-and-drop or the file picker (supports PNG, JPG, WEBP)
3. **Click Translate** — the pipeline will:
   - Detect text regions via Roboflow cloud API
   - Extract Japanese text with manga-ocr
   - Translate to English with Gemini
   - Clean the bubbles and render English text
4. **Browse results** with the side-by-side viewer (Original / Translated)
5. **Download** the translated pages

---

## Project Structure

```
Final Translator/
├── app.py                          # Flask web server (entry point)
├── .env                            # API keys (not tracked in git)
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates for web GUI
├── fonts/                          # Font files for text rendering
└── colorless_translator/           # Core package
    ├── config/                     # Settings and configuration
    ├── core/                       # Pipeline orchestration
    ├── detection/                  # Roboflow API + region merging
    ├── ocr/                        # manga-ocr wrapper
    ├── translation/                # Gemini translation + caching
    ├── rendering/                  # Inpainting + text rendering
    └── utils/                      # Helper utilities
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Upload      │────▶│  Roboflow    │────▶│  manga-ocr  │
│  Manga Page  │     │  Cloud API   │     │  (Local)    │
└─────────────┘     │  Detection   │     │  OCR        │
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │  Inpaint &   │◀────│  Gemini API │
                    │  Render Text │     │  Translation│
                    └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Translated │
                    │  Output     │
                    └─────────────┘
```

---

## Configuration

All detection and rendering behaviour can be tuned in `colorless_translator/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `conf_text_bubble` | `0.20` | Minimum detection confidence |
| `roboflow_min_conf` | `0.20` | Roboflow API confidence floor |
| `base_font_size` | `14` | Base font size for rendering |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'manga_ocr'`
Use Python 3.11. Other versions may have compatibility issues:
```bash
python --version  # should show 3.11.x
pip install manga-ocr
```

### `'charmap' codec can't encode characters`
This is a Windows console encoding issue. The app forces UTF-8 output automatically, but if it persists:
```bash
set PYTHONIOENCODING=utf-8
python app.py
```

### `CUDA not available, using CPU`
The app works fine on CPU. For GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### API quota exceeded
- Gemini free tier: 1,500 requests/day (resets at midnight Pacific Time)
- Roboflow free tier: 1,000 inferences/month
- Use translation cache to reduce repeated API calls

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | [Roboflow](https://roboflow.com/) Custom-Trained Model |
| OCR | [manga-ocr](https://github.com/kha-white/manga-ocr) |
| Translation | [Google Gemini 2.0](https://ai.google.dev/) |
| Inpainting | OpenCV + NumPy |
| Web GUI | Flask |
| Font Rendering | Pillow (PIL) |

---

## Credits

- **Roboflow** — Cloud inference platform for custom object detection
- **manga-ocr** by [kha-white](https://github.com/kha-white/manga-ocr) — Japanese manga OCR
- **Google Gemini** — AI translation engine
- **Ultralytics** — YOLOv8 model architecture

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Made for the manga community** ⭐
