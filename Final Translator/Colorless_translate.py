import os
import cv2
from manga_ocr import MangaOcr
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from yolov5 import YOLOv5
import google.generativeai as genai
import re

print("--- Professional Manga Translator Script (Definitive Version) ---")

# --- CONFIGURATION ---
FONT_PATH = r"C:\Users\Miggy\Documents\Final Translator\fonts\animeace2_reg.ttf"
FONT_SIZE = 28
FONT_SCALE_FACTOR = 0.9 # Changed from 1.1 to a more appropriate value.
INPUT_FOLDER = "manga pages"
OUTPUT_FOLDER = "output"
YOLO_MODEL_PATH = r"C:\Users\Miggy\Documents\Final Translator\best.pt"

# --- GEMINI API SETUP ---
API_KEY = ""  # <--- PASTE YOUR GOOGLE AI STUDIO KEY HERE
if API_KEY == "YOUR_API_KEY_HERE":
    print("FATAL ERROR: Please replace 'YOUR_API_KEY_HERE' with your API key."); exit()
genai.configure(api_key=API_KEY)
print("Google Gemini API configured successfully.")


# --- MODEL LOADING ---
print("Loading Models...")
mocr = MangaOcr()
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
bubble_model = YOLOv5(YOLO_MODEL_PATH)
print("All models loaded successfully.")


# --- HELPER FUNCTIONS ---
def detect_bubbles(image_cv):
    results = bubble_model.predict(image_cv)
    bubbles = []
    if len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
        for det in results.xyxy[0]:
            x, y, x2, y2 = map(int, det[:4]); bubbles.append((x, y, x2 - x, y2 - y))
    return bubbles

def draw_adaptive_text_in_box(draw, text, box, font_path, initial_font_size):
    """Draws text, shrinking font size until it fits the box."""
    x, y, w, h = box
    padding = 10 # Increased padding for a cleaner look
    if w <= padding * 2 or h <= padding * 2: return
    w -= padding * 2; h -= padding * 2
    x += padding; y += padding

    font_size = initial_font_size
    while font_size > 5:
        font = ImageFont.truetype(font_path, font_size)
        try: avg_char_width = font.getlength('A')
        except AttributeError: (width, _), avg_char_width = font.getsize("A"), width

        chars_per_line = max(1, w // avg_char_width if avg_char_width > 0 else 1)
        wrapped_text = textwrap.fill(text, width=int(chars_per_line))
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        if text_w < w and text_h < h: break
        font_size -= 1
    else:
        print(f"    Warning: Could not fit text properly. Final font size: {font_size}")

    final_text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
    final_w, final_h = final_text_bbox[2] - final_text_bbox[0], final_text_bbox[3] - final_text_bbox[1]
    text_x, text_y = x + (w - final_w) / 2, y + (h - final_h) / 2
    draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=(0, 0, 0), align="center")

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
    x_inter1, y_inter1 = max(x1, x2), max(y1, y2)
    x_inter2, y_inter2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area != 0 else 0

def sort_and_filter_bubbles(boxes, iou_threshold=0.8, y_tolerance_ratio=0.5):
    filtered_boxes = []
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    for box in boxes:
        if not any(calculate_iou(box, filtered_box) > iou_threshold for filtered_box in filtered_boxes):
            filtered_boxes.append(box)
    rows = []; remaining_boxes = sorted(filtered_boxes, key=lambda box: box[1])
    while remaining_boxes:
        base_box = remaining_boxes.pop(0); row = [base_box]
        y_center, y_tolerance = base_box[1] + base_box[3] / 2, base_box[3] * y_tolerance_ratio
        other_boxes = []
        for other in remaining_boxes:
            if abs(y_center - (other[1] + other[3] / 2)) <= y_tolerance: row.append(other)
            else: other_boxes.append(other)
        remaining_boxes = other_boxes
        row.sort(key=lambda box: box[0], reverse=True); rows.append(row)
    return [box for row in rows for box in row]

def analyze_text_and_get_properties(bubble_roi_cv):
    gray_roi = cv2.cvtColor(bubble_roi_cv, cv2.COLOR_BGR2GRAY)
    text_mask = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 4)
    bubble_pil = Image.fromarray(cv2.cvtColor(bubble_roi_cv, cv2.COLOR_BGR2RGB))
    japanese_text = mocr(bubble_pil)
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_heights = [cv2.boundingRect(c)[3] for c in contours]
    avg_char_height = np.mean([h for h in char_heights if h > 5]) if any(h > 5 for h in char_heights) else 0
    return japanese_text, avg_char_height, text_mask

def create_precise_bubble_mask(bubble_roi_cv):
    gray_roi = cv2.cvtColor(bubble_roi_cv, cv2.COLOR_BGR2GRAY)
    background_mask = cv2.inRange(gray_roi, 200, 255)
    contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    precise_mask = np.zeros_like(gray_roi)
    cv2.drawContours(precise_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return precise_mask

# --- TRANSLATION FUNCTIONS ---
def translate_entire_page(text_list):
    if not text_list: return []
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(text_list)])
    prompt = f"""You are an expert manga translator. Your task is to translate a list of Japanese text snippets from a single manga page, provided in the correct right-to-left reading order.\n\n## Instructions:\n1. Translate the snippets into natural, colloquial English, maintaining the context of the entire page.\n2. Your output MUST be a list of the English translations, separated by the special delimiter '|||'.\n3. The number of translations in your output must EXACTLY match the number of Japanese snippets.\n4. Do NOT include numbering (e.g., "1.") or commentary, only the translated text separated by '|||'.\n\n## Japanese Snippets:\n{numbered_texts}"""
    try:
        response = gemini_model.generate_content(prompt)
        translations = [t.strip() for t in response.text.split('|||')]
        if len(translations) == len(text_list): return translations
        else: return [translate_text_gemini(text) for text in text_list]
    except Exception: return [translate_text_gemini(text) for text in text_list]
def translate_text_gemini(text):
    prompt = f"Translate to natural English: \"{text}\""
    try: response = gemini_model.generate_content(prompt); return response.text.strip()
    except Exception: return "[Translation Error]"

# --- CORE PROCESSING LOGIC ---
def process_image(image_path, output_path):
    print(f"\nProcessing image: {image_path}")
    original_cv = cv2.imread(image_path)
    if original_cv is None: print(f"❌ Could not read image: {image_path}"); return

    # Pass 1: Analyze
    print("--- Pass 1: Analyzing Bubbles ---")
    all_bubbles = detect_bubbles(original_cv)
    all_bubbles = sort_and_filter_bubbles(all_bubbles)
    if not all_bubbles: print(" -> No bubbles found."); cv2.imwrite(output_path, original_cv); return
    page_data = []
    for i, box in enumerate(all_bubbles):
        japanese_text, avg_jp_height, text_mask = analyze_text_and_get_properties(original_cv[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
        if not re.search(r'[ぁ-んァ-ン一-龯]', japanese_text): continue
        print(f" -> Found Block #{i+1}: '{japanese_text}' (Avg Char Height: {avg_jp_height:.1f}px)")
        page_data.append({'box': box, 'text': japanese_text, 'jp_height': avg_jp_height, 'text_mask': text_mask})
    if not page_data: print(" -> No valid text found."); cv2.imwrite(output_path, original_cv); return

    # Translation
    print("\n--- Translating Entire Page ---")
    english_translations = translate_entire_page([item['text'] for item in page_data])
    for i, item in enumerate(page_data): item['english'] = english_translations[i]

    # Pass 2: Inpaint and Typeset
    print("\n--- Pass 2: Inpainting and Typesetting ---")
    final_cv = original_cv.copy()
    for i, item in enumerate(page_data):
        print(f" -> Cleaning & Drawing Block #{i+1}: \"{item['english']}\"")
        x, y, w, h = item['box']
        bubble_roi = final_cv[y:y+h, x:x+w]

        precise_bubble_mask = create_precise_bubble_mask(bubble_roi)
        if precise_bubble_mask is None: precise_bubble_mask = 255

        inpaint_mask = cv2.bitwise_and(item['text_mask'], precise_bubble_mask)
        # Use a slightly larger kernel for more effective cleaning of text "ghosts"
        kernel = np.ones((3,3), np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)

        healed_roi = cv2.inpaint(bubble_roi, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        final_cv[y:y+h, x:x+w] = healed_roi

    final_pil = Image.fromarray(cv2.cvtColor(final_cv, cv2.COLOR_BGR2RGB))
    final_draw = ImageDraw.Draw(final_pil)
    for item in page_data:
        initial_font_size = int(item['jp_height'] * FONT_SCALE_FACTOR) if item['jp_height'] > 0 else FONT_SIZE
        draw_adaptive_text_in_box(final_draw, item['english'], item['box'], FONT_PATH, initial_font_size)

    final_pil.save(output_path, "PNG")
    print(f"\n✅ Finished and saved: {output_path}")

# --- MAIN EXECUTION ---
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            process_image(os.path.join(INPUT_FOLDER, filename), os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.png"))

if __name__ == "__main__":
    main()
    print("\n--- Script Finished ---")