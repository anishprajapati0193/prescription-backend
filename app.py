# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io, re
from PIL import Image
import numpy as np
import cv2
import pytesseract
from rapidfuzz import process, fuzz

app = FastAPI()

# CORS for development - change in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Medicine DB (canonical keys) ---
MEDICINE_DB = {
    "paracetamol": {"display": "Paracetamol", "price": 50, "aliases": ["paracetamol", "crocin", "dolo"]},
    "amoxicillin": {"display": "Amoxicillin", "price": 100, "aliases": ["amoxicillin", "amox"]},
    "cetirizine": {"display": "Cetirizine", "price": 40, "aliases": ["cetirizine", "cetrizine"]},
    "ibuprofen": {"display": "Ibuprofen", "price": 60, "aliases": ["ibuprofen", "brufen"]},
}

# build alias -> canonical map and candidate list for fuzzy match
CANDIDATE_TO_CANON = {}
for canon, info in MEDICINE_DB.items():
    for alias in info["aliases"]:
        CANDIDATE_TO_CANON[alias.lower()] = canon
CANDIDATES = list(CANDIDATE_TO_CANON.keys())

# If you're on Windows and tesseract is not on PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_imagefile(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img

def preprocess_pil(pil_img: Image.Image) -> Image.Image:
    """Basic preprocessing to improve OCR: grayscale, resize (if small), gaussian blur, binary threshold."""
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # upscale small images to improve OCR
    h, w = gray.shape
    scale = 1
    if max(h, w) < 1000:
        scale = 2
    if scale != 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # denoise and threshold
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(th)

def ocr_extract(pil_img: Image.Image) -> str:
    # psm 6 works well for blocks of text; adjust if needed
    config = "--psm 6"
    text = pytesseract.image_to_string(pil_img, config=config)
    text = re.sub(r'\r\n', '\n', text)             # unify newlines
    text = re.sub(r'[ \t]+', ' ', text)           # collapse repeated spaces/tabs
    return text.strip()

def find_patient_id(text: str):
    # Try common labels first
    patterns = [
        r'patient[:\s]*id[:\s]*([A-Za-z0-9\-_/]+)',
        r'patient[:\s]*no[:\s]*([A-Za-z0-9\-_/]+)',
        r'uhid[:\s]*([A-Za-z0-9\-_/]+)',
        r'mrn[:\s]*([A-Za-z0-9\-_/]+)',
        r'reg(?:istration)?\s*no[:\s]*([A-Za-z0-9\-_/]+)'
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1)
    # fallback: any long sequence of digits (5-12 digits)
    m = re.search(r'\b(\d{5,12})\b', text)
    if m:
        return m.group(1)
    return None

def extract_medicines_from_text(text: str):
    """Parse text line-by-line. For each line try fuzzy match to known medicine aliases.
       If match -> available True and we report price and qty.
       If not matched -> create an unavailable entry (show to user)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results = []
    seen_keys = set()

    for line in lines:
        # find quantity tokens in the line: "2 tabs", "10 ml", or "2x10"
        qty = None
        # patterns: '2 tabs', '2 x 10' (we take first number as qty), or simply standalone number
        m_qty = re.search(r'(\d+)\s*(?:x|\*|X)?\s*(\d+)?\s*(tabs?|tablets?|caps?|ml|mg|strip|syrup)?', line, re.I)
        if m_qty:
            # many prescriptions write "2 x 10" meaning 2 strips of 10; taking the first number as quantity to dispense
            qty = int(m_qty.group(1))

        # best fuzzy alias in this line
        best = process.extractOne(line, CANDIDATES, scorer=fuzz.token_sort_ratio)
        if best:
            candidate_alias, score, _ = best
        else:
            candidate_alias, score = None, 0

        if score and score >= 65:
            canon = CANDIDATE_TO_CANON[candidate_alias]
            key = canon
            if key in seen_keys:
                continue  # skip duplicates
            seen_keys.add(key)

            if qty is None:
                # try to detect a number after the alias in the same line
                m_after = re.search(rf'{re.escape(candidate_alias)}.*?(\d+)', line, re.I)
                qty = int(m_after.group(1)) if m_after else 1

            unit_price = MEDICINE_DB[canon]['price']
            results.append({
                "canonical": canon,
                "display": MEDICINE_DB[canon]['display'],
                "name_raw": line,
                "quantity": qty,
                "unit_price": unit_price,
                "line_total": qty * unit_price,
                "available": True
            })
        else:
            # heuristic: if line contains letters (likely medicine name), add as unavailable so user sees it
            if re.search(r'[A-Za-z]{3,}', line):
                key = line.lower()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if qty is None:
                    m_any = re.search(r'(\d+)', line)
                    qty = int(m_any.group(1)) if m_any else 1
                results.append({
                    "canonical": None,
                    "display": line,
                    "name_raw": line,
                    "quantity": qty,
                    "unit_price": 0,
                    "line_total": 0,
                    "available": False
                })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pil = read_imagefile(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    pre = preprocess_pil(pil)
    text = ocr_extract(pre)
    patient_id = find_patient_id(text)
    medicines = extract_medicines_from_text(text)
    grand_total = sum(m['line_total'] for m in medicines if m['available'])

    return JSONResponse({
        "ocr_text": text,
        "patient_id": patient_id,
        "detected_medicines": medicines,
        "grand_total": grand_total
    })
