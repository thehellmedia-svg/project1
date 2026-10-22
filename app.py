import json
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st


# ---------- extraction rules ----------

import re


def _first(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip(" \t:#-")
    return None


def _number(value):
    if value is None:
        return None
    cleaned = value.replace(",", "").replace("₹", "").replace("$", "").replace("€", "").strip()
    try:
        # Return a JSON-friendly number, preserving decimal precision as a string where useful.
        return str(Decimal(cleaned))
    except InvalidOperation:
        return value.strip()


def _extract_items(text):
    """Conservative row parser for common text rows:
       Item name | Qty | Unit price | Amount
       or Item name   Qty   Unit price   Amount
    """
    items = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) < 5:
            continue
        if re.search(r"item|description|quantity|qty|unit price|amount|subtotal|total", line, re.I):
            continue

        # Pipe/comma-separated rows are more reliable than arbitrary whitespace.
        parts = [p.strip() for p in re.split(r"\s*\|\s*", line)]
        if len(parts) >= 4:
            name, qty, price, amount = parts[0], parts[1], parts[2], parts[3]
            if re.fullmatch(r"\d+(?:\.\d+)?", qty.replace(",", "")):
                items.append({
                    "name": name,
                    "quantity": _number(qty),
                    "unit_price": _number(price),
                    "line_total": _number(amount),
                    "extraction_method": "pipe-separated row heuristic",
                })
                continue

        # Tabular text may have been extracted with variable spaces.
        match = re.match(
            r"^(.+?)\s{2,}(\d+(?:,\d{3})*(?:\.\d+)?)\s{2,}"
            r"([₹$€]?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?)\s{2,}"
            r"([₹$€]?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?)$",
            line,
        )
        if match:
            items.append({
                "name": match.group(1).strip(),
                "quantity": _number(match.group(2)),
                "unit_price": _number(match.group(3)),
                "line_total": _number(match.group(4)),
                "extraction_method": "whitespace table heuristic",
            })
    return items


def extract_fields(text: str, document_type: str) -> tuple[dict, list[str]]:
    warnings = []
    if not text.strip():
        return {
            "document_id": None,
            "vendor": None,
            "document_date": None,
            "currency": None,
            "items": [],
            "subtotal": None,
            "tax": None,
            "total": None,
        }, ["No text available for field extraction."]

    document_id = _first([
        r"(?:invoice\s*(?:number|no\.?|#)|invoice\s*id)\s*[:#-]?\s*([A-Z0-9/-]+)",
        r"(?:purchase\s*order|PO)\s*(?:number|no\.?|#|id)?\s*[:#-]?\s*([A-Z0-9/-]+)",
        r"(?:GRN|delivery\s*receipt)\s*(?:number|no\.?|#|id)?\s*[:#-]?\s*([A-Z0-9/-]+)",
    ], text)

    vendor = _first([
        r"(?:vendor|supplier|seller|billed\s*by)\s*[:\-]\s*(.+)",
    ], text)

    date = _first([
        r"(?:invoice\s*date|document\s*date|date)\s*[:\-]\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})",
    ], text)

    currency = "INR" if "₹" in text or re.search(r"\bINR\b|\bRs\.?", text, re.I) else None
    items = _extract_items(text)

    subtotal = _first([
        r"(?:sub\s*total|subtotal)\s*[:\-]?\s*[₹$€]?\s*([\d,]+(?:\.\d{1,2})?)",
    ], text)
    tax = _first([
        r"(?:tax|GST|VAT)\s*[:\-]?\s*[₹$€]?\s*([\d,]+(?:\.\d{1,2})?)",
    ], text)
    total = _first([
        r"\b(?:grand\s*total|invoice\s*total|total\s*due|total)\s*[:\-]?\s*[₹$€]?\s*([\d,]+(?:\.\d{1,2})?)",
    ], text)

    if not document_id:
        warnings.append("Document ID was not detected. Check the text and edit the JSON if needed.")
    if not vendor:
        warnings.append("Vendor/supplier was not detected; templates may use a different label.")
    if not items:
        warnings.append(
            "No line items were detected by the starter table rules. This is expected for many layouts; "
            "inspect raw text and edit the JSON, or improve the table parser for your dataset."
        )

    data = {
        "document_id": document_id,
        "vendor": vendor,
        "document_date": date,
        "currency": currency,
        "items": items,
        "subtotal": _number(subtotal),
        "tax": _number(tax),
        "total": _number(total),
    }
    return data, warnings

# ---------- optional OCR ----------


def ocr_image_or_pdf(path: str) -> str:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Run: pip install -r requirements-ocr.txt"
        ) from exc

    # PaddleOCR API differs slightly by version. This uses the current common API.
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    result = ocr.predict(path)

    lines = []
    def walk(obj):
        if isinstance(obj, dict):
            # Newer result objects may contain recognized text in rec_texts.
            rec_texts = obj.get("rec_texts")
            if isinstance(rec_texts, list):
                lines.extend(str(x) for x in rec_texts)
            for key, value in obj.items():
                if key != "rec_texts":
                    walk(value)
        elif isinstance(obj, (list, tuple)):
            # Older-style output can be nested [box, (text, score)].
            if len(obj) == 2 and isinstance(obj[1], (list, tuple)) and obj[1] and isinstance(obj[1][0], str):
                lines.append(obj[1][0])
            else:
                for value in obj:
                    walk(value)
        elif hasattr(obj, "json"):
            try:
                walk(obj.json)
            except Exception:
                pass

    walk(result)
    # Preserve first occurrence order while removing duplicates.
    seen = set()
    unique = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique.append(line)
    return "\n".join(unique)

# ---------- pipeline ----------

from pathlib import Path
import re
import tempfile
import os




def process_document(file_bytes: bytes, filename: str, document_type: str) -> dict:
    """Extract text from a PDF/image, then map it into a simple JSON schema."""
    suffix = Path(filename).suffix.lower()
    raw_text = ""
    text_source = "PyMuPDF (digital PDF)"
    warnings = []

    if suffix == ".pdf":
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                raw_text = "\n".join(page.get_text("text") for page in pdf)
        except Exception as exc:
            warnings.append(f"PDF text reading failed: {exc}")

        # Scanned PDFs commonly contain little/no selectable text.
        if len(re.sub(r"\s+", "", raw_text)) < 30:
            text_source = "PaddleOCR fallback"
            raw_text = _run_ocr(file_bytes, suffix, warnings)
    else:
        text_source = "PaddleOCR"
        raw_text = _run_ocr(file_bytes, suffix, warnings)

    if not raw_text.strip():
        warnings.append(
            "No usable text was detected. Check the file quality or install/configure PaddleOCR."
        )

    data, field_warnings = extract_fields(raw_text, document_type)
    warnings.extend(field_warnings)
    data["source_filename"] = filename
    data["document_type"] = document_type

    return {
        "filename": filename,
        "text_source": text_source,
        "raw_text": raw_text,
        "data": data,
        "warnings": warnings,
    }


def _run_ocr(file_bytes: bytes, suffix: str, warnings: list[str]) -> str:
    # OCR dependency is optional so digital-PDF mode can run without it.
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(file_bytes)
            path = temp.name
        try:
            return ocr_image_or_pdf(path)
        finally:
            os.unlink(path)
    except Exception as exc:
        warnings.append(
            "OCR could not run. Install the optional OCR dependencies listed in requirements.txt. "
            f"Details: {exc}"
        )
        return ""

# ---------- app ----------




st.set_page_config(page_title="Invoice Document Extractor", page_icon="📄", layout="wide")

# ---- Light styling: restrained accent, soft cards, tidy spacing ----
st.markdown(
    """
    <style>
    :root { --accent: #6366f1; --accent-2: #06b6d4; }
    .block-container { padding-top: 2.2rem; max-width: 1100px; }
    .hero h1 {
        font-size: 2.1rem; font-weight: 800; margin: 0 0 .25rem 0;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { opacity: .75; margin: 0 0 1.2rem 0; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
        padding: .8rem 1rem; background: rgba(99,102,241,.05);
    }
    .field-card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
        padding: .7rem 1rem; margin-bottom: .6rem;
        border-left: 4px solid var(--accent);
    }
    .field-card .lbl { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; }
    .field-card .val { font-size: 1.05rem; font-weight: 600; }
    .totals { border-radius: 12px; padding: .8rem 1rem; background: rgba(6,182,212,.07);
              border: 1px solid rgba(6,182,212,.3); }
    .totals div { display: flex; justify-content: space-between; padding: .15rem 0; }
    .totals .grand { border-top: 1px solid rgba(128,128,128,.3); margin-top: .35rem;
                     padding-top: .45rem; font-weight: 800; font-size: 1.1rem; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2)); border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>📄 Document Extractor</h1>'
    "<p>Upload a Purchase Order, Invoice, or Delivery Receipt. Get structured data back.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Read digital PDF text with **PyMuPDF**.\n"
        "2. If the PDF has no usable text, try **PaddleOCR**.\n"
        "3. Extract fields using rules and table-row patterns.\n"
        "4. Validate basic fields and export JSON."
    )
    st.caption("This is a prototype. Always review extracted values.")

DOC_TYPES = ["Invoice", "Purchase Order", "Delivery Receipt / GRN"]
doc_type = st.radio("Document type", DOC_TYPES, horizontal=True)
uploaded = st.file_uploader("Upload PDF, PNG, JPG or JPEG", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    raw = uploaded.getvalue()
    st.caption(f"📎 {uploaded.name} · {len(raw):,} bytes")
    if st.button("Extract information", type="primary"):
        with st.spinner("Reading document and extracting fields..."):
            try:
                st.session_state["result"] = process_document(raw, uploaded.name, doc_type)
            except Exception as e:
                st.error(f"Could not process this file: {e}")
                st.stop()


def field_card(label: str, value) -> str:
    return (
        f'<div class="field-card"><div class="lbl">{label}</div>'
        f'<div class="val">{value or "—"}</div></div>'
    )


if "result" in st.session_state:
    result = st.session_state["result"]
    data = result["data"]
    cur = data.get("currency") or ""

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Document type", data.get("document_type", "—"))
    c2.metric("Text source", result["text_source"])
    c3.metric("Characters extracted", f'{len(result["raw_text"]):,}')

    for warning in result["warnings"]:
        st.warning(warning)

    tab_doc, tab_json, tab_raw = st.tabs(["Document view", "Edit JSON", "Raw text"])

    with tab_doc:
        a, b, c = st.columns(3)
        a.markdown(field_card("Document ID", data.get("document_id")), unsafe_allow_html=True)
        b.markdown(field_card("Vendor", data.get("vendor")), unsafe_allow_html=True)
        c.markdown(field_card("Date", data.get("document_date")), unsafe_allow_html=True)

        items = data.get("items") or []
        if items:
            df = pd.DataFrame(items).drop(columns=["extraction_method"], errors="ignore")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No line items detected.")

        st.markdown(
            f'<div class="totals">'
            f'<div><span>Subtotal</span><span>{cur} {data.get("subtotal") or "—"}</span></div>'
            f'<div><span>Tax</span><span>{cur} {data.get("tax") or "—"}</span></div>'
            f'<div class="grand"><span>Total</span><span>{cur} {data.get("total") or "—"}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with tab_json:
        edited = st.text_area(
            "Correct any OCR/extraction errors here",
            value=json.dumps(data, indent=2, ensure_ascii=False),
            height=360,
        )
        try:
            parsed = json.loads(edited)
            st.download_button(
                "Download structured JSON",
                data=json.dumps(parsed, indent=2, ensure_ascii=False),
                file_name=f'{Path(result["filename"]).stem}_extracted.json',
                mime="application/json",
            )
        except json.JSONDecodeError as e:
            st.error(f"JSON is not valid yet: {e}")

    with tab_raw:
        st.text(result["raw_text"] or "(No text could be extracted.)")