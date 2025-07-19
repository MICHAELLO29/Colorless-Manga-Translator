# 🎨 Colorless-Manga-Translator

> **Automate manga translation from Japanese to English with AI — from bubble detection to typeset output!**

---

> ⚠️ **Reminder:**
> - This tool is designed for colorless (black-and-white) manga. For colored manga, it will only translate the text if the speech bubble or the background of the text is white. If the background is not white, detection and inpainting may not work correctly.
> - Some translated text may become larger or overlap inside the speech bubbles, especially if the English translation is much longer than the original Japanese text. Manual adjustment may be needed for perfect results.
> - Use this script in terminal if the run button didnt work in vs code:
```bash
   cd "Final Translator"
   py -3.11 Colorless_translate.py
   ```
---

## ✨ What is Colorless-Manga-Translator?

> **Colorless-Manga-Translator** is your all-in-one, open-source toolkit for turning raw Japanese manga pages into beautifully typeset English versions — automatically! Whether you're a manga fan, scanlator, or researcher, this project saves you hours of manual work and delivers professional results.

---

## 🚀 **How It Works: The Magic Pipeline**

Follow these steps to understand how the Colorless-Manga-Translator processes your manga pages:

1. **🖼️ Input Manga Page**
   - Place your black-and-white manga image(s) in the `Final Translator/manga pages/` folder.

2. **🔍 Detect Speech Bubbles (YOLOv5)**
   - The tool uses a custom-trained YOLOv5 model to automatically find and outline all speech bubbles in each manga page.

3. **📝 Extract Japanese Text (Manga OCR)**
   - For each detected bubble, the script uses Manga OCR to read and extract the Japanese text, even if it's handwritten or stylized.

4. **🌐 Translate to English (Google Gemini API)**
   - The extracted Japanese text is sent to the Google Gemini API, which returns a natural, context-aware English translation.

5. **🧹 Clean Bubble (Inpainting)**
   - The original Japanese text is erased from the bubble using inpainting techniques, leaving a clean white space for the new text.

6. **✍️ Typeset English Text**
   - The translated English text is drawn into the cleaned bubble using a manga-style font, aiming for a natural and readable look.

7. **📄 Output: Translated Manga Page**
   - The final, fully translated and typeset manga page is saved in the `Final Translator/output/` folder, ready to read or share!

---

## 🏆 **Why You'll Love This Project**

- **⚡ Fully Automated:** Just drop your manga images in, and get translated, typeset pages out!
- **🎯 High Accuracy:** Custom-trained YOLOv5 model for manga speech bubbles.
- **🈶 Specialized OCR:** Extracts even handwritten or stylized Japanese text.
- **💬 Natural Translations:** Uses Google Gemini AI for context-aware, colloquial English.
- **🖋️ Professional Typesetting:** Clean, readable, manga-style fonts.
- **🔓 100% Free & Open Source:** No paywalls, no locked features.
- **🔧 Extensible:** Tweak, retrain, or swap components as you wish.

---

## 📦 **Folder Structure at a Glance**

| Folder/File | Purpose |
|-------------|---------|
| `Final Translator/Colorless_translate.py` | Main script: run this to translate manga |
| `Final Translator/manga pages/` | Put your input manga images here |
| `Final Translator/output/` | Translated, typeset manga pages appear here |
| `Final Translator/fonts/` | Fonts for typesetting (e.g., animeace2_reg.ttf) |
| `Final Translator/best.pt` | YOLOv5 model weights for bubble detection |
| `Final Translator/Manga Speech Bubble Detection.v3i.yolov5pytorch/` | Training code, dataset, and YOLOv5 files |

---

## 🛠️ **Get Started in 5 Easy Steps**

> ⚠️ **Note:**
> - This tool is intended for black-and-white manga. For colored manga, it will only work if the speech bubble or text background is white.
> - Some translated text may become larger or overlap inside the bubbles. Manual adjustment may be needed for best results.
> - I use python 3.11.x for the script to work. Some of the libraries are not compatible in the new python version.
> - I tried to use it more in some manga pages and I see that the best font size is 14 whether the japanese texts are long it will fit inside the speech bubble text if its converted to english. Well you can adjust it to you font size. 

1. **Clone the Repository Locally**
   ```bash
   git clone https://github.com/MICHAELLO29/Colorless-Manga-Translator.git
   cd Colorless-Manga-Translator
   ```
2. **Install Requirements**
   ```bash
   pip install -r "Final Translator/Manga Speech Bubble Detection.v3i.yolov5pytorch/yolov5/requirements.txt"
   pip install manga-ocr google-generativeai
   ```
3. **Get Your Free Google Gemini API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in, create an API key, and paste it into `Colorless_translate.py`:
     ```python
     API_KEY = "YOUR_API_KEY_HERE"
     ```
4. **Add Manga Images**
   - Place your manga pages in `Final Translator/manga pages/`
5. **Run the Translator!**
   ```bash
   python Final Translator/Colorless_translate.py
   ```
   - Your translated pages will appear in `Final Translator/output/`

---

> **Sample Provided:**
> - I created a sample manga image for you to try out! You can find it in the `Final Translator/manga pages/` folder. After running the translator, the output of this sample can be seen in the `Final Translator/output/` folder.
>
>   ![Original Manga Page](Final%20Translator/manga%20pages/1.png)
>
>   ![Translated Manga Page](Final%20Translator/output/1.png)
---

## 💡 **How Does Each Step Work?**

- **Speech Bubble Detection:**
  - YOLOv5 finds all speech bubbles, even in complex layouts.
- **Japanese Text Extraction:**
  - Manga OCR reads both printed and handwritten Japanese text.
- **Translation:**
  - Google Gemini API translates with context, not just word-for-word.
- **Typesetting:**
  - The script erases the original text and redraws the English translation in a manga-style font.

---

## 🧠 **About the Free API**

- **Google Gemini API** is used for translation.
- You can use it for free (with usage limits) by creating a Google account and generating an API key.
- No paid subscription required for basic use!

---

## 📚 **Credits & Licenses**

- **YOLOv5:** [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (AGPL-3.0)
- **Manga OCR:** [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr)
- **Google Gemini API:** [google-generativeai](https://github.com/google/generative-ai-python)
- **Dataset:** Provided by Roboflow user ([details](Final%20Translator/Manga%20Speech%20Bubble%20Detection.v3i.yolov5pytorch/README.dataset.txt))

---

## 🔮 **Future Work**

- **User-Friendly GUI or Website:**
  - I plan to develop a graphical user interface (GUI) or a web-based platform that allows users to simply drag and drop their manga images for translation. This will make the tool even more portable and hassle-free, removing the need to use command-line scripts. The goal is to make manga translation accessible to everyone, regardless of technical background—just upload your images and get instant, beautifully typeset English manga pages!


