# Quick Start Guide - Web GUI

## 🚀 Getting Started (3 Steps)

### Step 1: Install Dependencies

```bash
cd "Final Translator"
pip install -r requirements.txt
```

### Step 2: Configure API Key

You can set your API key in **two ways**:

**Option A:** Edit the `.env` file directly:
```bash
GEMINI_API_KEY=AIzaSyD...
```

**Option B:** Enter it in the web GUI (recommended — the Settings panel will prompt you on first launch).

**Get a free key:** https://aistudio.google.com/apikey

---

### Step 3: Launch the Web GUI

```bash
cd "Final Translator"
python app.py
```

**Wait for the startup message:**
```
============================================================
  Colorless Manga Translator — Web GUI
============================================================

  Starting web server...
  Open your browser to: http://localhost:7860
  Press Ctrl+C to stop
```

Open **http://localhost:7860** in your browser.

---

### Step 4: Translate!

1. **Enter your API key** via the Settings panel (if not already in `.env`)
2. **Upload** a manga page (drag-and-drop or click to browse)
3. **Choose** a strategy (Auto recommended)
4. **Click** 🌐 Translate
5. **Wait** 30-60s (first run loads AI models)
6. **View** the side-by-side original vs. translated result
7. **Download** the translated image

---

## ❌ Common Errors & Solutions

### Error: "No API Key" badge in header

**Solution:**
1. Click the **⚙ Settings** button in the header
2. Paste your API key into the input field
3. Click **Test & Save**
4. The badge should turn green once validated

---

### Error: "Translator not available"

**This means model loading failed.** Common causes:
- Missing YOLO model file (`models/dabest.pt`)
- Missing font file (`fonts/animeace2_reg.ttf`)
- Python packages not installed (`pip install -r requirements.txt`)

Check the terminal output for details.

---

### Error: "API quota exhausted"

**Solution:**
- Wait for the free-tier quota to reset (usually resets daily)
- Or upgrade to a paid API tier at [Google AI Studio](https://aistudio.google.com)

---

## 🎯 Features

### Single Page Translation
- Upload one page
- Side-by-side comparison (original vs. translated)
- Download translated image
- View stats (strategy used, blocks translated, time)

### Batch Processing
- Upload multiple pages at once
- Process sequentially

### Strategy Selection
- **🤖 Auto Detect** — Automatically picks the best strategy
- **⚡ Action** — Punchy, brief text for action scenes
- **💬 Dialogue** — Natural conversation flow
- **📄 Standard** — Balanced general-purpose

### Translation Caching
- Translations are cached to `translation_cache.json`
- Repeated text is instant (0 API calls)
- Cache persists across sessions

---

## ⚡ Tips

### 1. Keep the Server Running
- Models stay loaded in memory
- Subsequent translations are much faster
- No reload penalty

### 2. Use Auto Strategy
- Automatically adapts to page content
- Best default choice for most pages

### 3. Check the Terminal
- Detailed logs appear in the terminal where `app.py` is running
- Useful for debugging translation issues

---

## 📊 Expected Performance

| Task | Time |
|------|------|
| GUI Startup | Instant |
| First Translation (model loading) | 30-60s |
| Subsequent Translations | 10-30s |
| Cached Translation | < 1s |

---

## ✅ Checklist

Before translating:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] API key configured (via `.env` or Settings panel)
- [ ] `app.py` running (see terminal output)
- [ ] Browser open to http://localhost:7860
- [ ] Status badge shows "API Ready" (green)
- [ ] Image uploaded
- [ ] Strategy selected

---

## 🎉 You're Ready!

**Start translating manga!** 🌟
