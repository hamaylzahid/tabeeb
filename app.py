# =========================================
#  TabeebAI — Unified Streamlit App
#  AI Medical Triage 
# =========================================

import os
import json
import time
import base64
import tempfile
import traceback
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================================
#  GROQ CLIENT
# =========================================
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

# =========================================
#  DISEASE DATASET
# =========================================
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "diseases_symptoms.csv")
disease_df = None
_SYMPTOM_COLS = []

try:
    disease_df = pd.read_csv(_DATA_PATH)
    disease_df.columns = (
        disease_df.columns.str.lower().str.strip().str.replace(" ", "_")
    )
    _SYMPTOM_COLS = [c for c in disease_df.columns if c != "disease"]
    print(f"[TabeebAI] Disease dataset: {len(disease_df)} diseases, {len(_SYMPTOM_COLS)} symptoms")
except Exception as e:
    print(f"[TabeebAI] Disease dataset not found: {e}")


def _match_symptom_to_column(symptom_name: str):
    query_words = set(
        symptom_name.lower().replace("_", " ").replace("-", " ").split()
    )
    stop_words = {"severe", "mild", "moderate", "acute", "chronic",
                  "sudden", "persistent", "intense", "sharp", "dull"}
    query_words -= stop_words
    best_col, best_score = None, 0
    for col in _SYMPTOM_COLS:
        col_words = set(col.replace("_", " ").split())
        overlap = len(query_words & col_words)
        if overlap > best_score:
            best_score, best_col = overlap, col
    return best_col if best_score > 0 else None


def lookup_diseases(symptoms_list: list, top_n: int = 5) -> list:
    if disease_df is None or not symptoms_list:
        return []
    matched_cols = []
    for s in symptoms_list:
        col = _match_symptom_to_column(s.get("name", ""))
        if col and col not in matched_cols:
            matched_cols.append(col)
    if not matched_cols:
        return []
    scores = disease_df[matched_cols].sum(axis=1)
    total_cols = len(matched_cols)
    results = (
        disease_df[["disease"]]
        .assign(matched=scores, total=total_cols)
        .query("matched > 0")
        .sort_values("matched", ascending=False)
        .head(top_n)
    )
    return results.to_dict(orient="records")


# =========================================
#  RAG SETUP
# =========================================
_rag_model = None
_rag_embeddings = None
_rag_docs = []


def setup_rag():
    global _rag_model, _rag_embeddings, _rag_docs
    knowledge_path = os.path.join(os.path.dirname(__file__), "data", "medical_knowledge.json")
    try:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            _rag_docs = json.load(f)
        print("[TabeebAI] Loading embedding model...")
        _rag_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{d['title']}. {d['text']}" for d in _rag_docs]
        _rag_embeddings = _rag_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        print(f"[TabeebAI] RAG ready: {len(_rag_docs)} documents embedded")
    except Exception as e:
        print(f"[TabeebAI] RAG setup failed: {e}")


def retrieve_knowledge(query: str, n: int = 3) -> list:
    if _rag_model is None or _rag_embeddings is None or not _rag_docs:
        return []
    query_emb = _rag_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores = np.dot(_rag_embeddings, query_emb.T).flatten()
    top_idx = scores.argsort()[-n:][::-1]
    return [
        {"title": _rag_docs[i]["title"], "text": _rag_docs[i]["text"], "score": float(scores[i])}
        for i in top_idx if scores[i] > 0.2
    ]


setup_rag()

# =========================================
#  LANGUAGE DETECTION
# =========================================
def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097f')
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "unknown"
    if urdu_chars / total_alpha > 0.3:
        return "ur"
    if latin_chars / total_alpha > 0.5:
        return "en"
    return "other"


# =========================================
#  TRANSCRIPTION
# =========================================
def _call_whisper(audio_path: str, language=None):
    with open(audio_path, "rb") as f:
        kwargs = dict(file=f, model="whisper-large-v3-turbo")
        if language:
            kwargs["language"] = language
        response = client.audio.transcriptions.create(**kwargs)
    text = response.text.strip()
    return text, detect_language(text)


def transcribe_audio_file(audio_path: str):
    text, lang = _call_whisper(audio_path)
    if lang == "other":
        text, lang = _call_whisper(audio_path, language="ur")
    if lang == "other":
        return "Unsupported language detected. Please speak in Urdu or English only.", "unknown"
    return text, lang


# =========================================
#  TRANSLATION
# =========================================
def translate_if_needed(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    "Translate the following Urdu medical text into clear English. "
                    "Return only the translation, no explanations:\n\n" + text
                )
            }],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Translation Error: {str(e)}"


# =========================================
#  CLASSIFICATION
# =========================================
def classify_query(text: str) -> str:
    prompt = f"""You are a medical query classifier for a clinical triage system.
Classify the following patient statement into exactly ONE category:
MEDICAL     — describes any symptom, pain, illness, injury, medication, or health concern
CRISIS      — mentions self-harm, suicide, wanting to die, or harming others
NON_MEDICAL — anything unrelated to health (greetings, general questions, jokes, etc.)
Statement: "{text}"
Reply with ONLY one word: MEDICAL, CRISIS, or NON_MEDICAL"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        if "CRISIS" in result:
            return "crisis"
        if "NON_MEDICAL" in result or "NON" in result:
            return "non_medical"
        return "medical"
    except Exception:
        return "medical"


# =========================================
#  SYMPTOM EXTRACTION
# =========================================
def extract_symptoms(text: str) -> dict:
    prompt = f"""You are a clinical AI assistant.
STRICT RULES:
- Output MUST be ONLY in English
- Return ONLY valid JSON, no extra text, no markdown
Extract structured medical symptoms from this text:
{text}
FORMAT:
{{
  "chief_complaint": "",
  "symptoms": [
    {{
      "name": "",
      "severity": "mild/moderate/severe",
      "duration": ""
    }}
  ],
  "possible_conditions": [],
  "urgency": "low/medium/high",
  "language": "English"
}}"""
    output = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response.choices[0].message.content
        output = output.replace("```json", "").replace("```", "").strip()
        return json.loads(output)
    except Exception as e:
        return {"error": "Parsing failed", "details": str(e), "raw_output": output}


# =========================================
#  RISK SCORING
# =========================================
def calculate_risk_score(analysis: dict) -> int:
    if not analysis or "error" in analysis:
        return 0
    score = 0
    urgency_scores = {"low": 10, "medium": 40, "high": 75}
    score += urgency_scores.get(analysis.get("urgency", "low").lower(), 10)
    severity_bonus = {"mild": 3, "moderate": 8, "severe": 18}
    for symptom in analysis.get("symptoms", []):
        score += severity_bonus.get(symptom.get("severity", "mild").lower(), 3)
    emergency_keywords = [
        "chest pain", "difficulty breathing", "shortness of breath",
        "unconscious", "unresponsive", "severe bleeding", "heart attack",
        "stroke", "choking", "not breathing", "no pulse",
        "سینے میں درد", "سانس لینے میں دشواری", "بے ہوش"
    ]
    full_text = analysis.get("chief_complaint", "").lower()
    full_text += " " + " ".join([s.get("name", "") for s in analysis.get("symptoms", [])])
    for keyword in emergency_keywords:
        if keyword in full_text:
            score += 30
            break
    score += min(len(analysis.get("symptoms", [])) * 3, 15)
    return min(score, 100)


def get_risk_level(score: int) -> str:
    if score >= 71:
        return "RED"
    elif score >= 31:
        return "YELLOW"
    else:
        return "GREEN"


# =========================================
#  SOAP REPORT
# =========================================
def generate_soap_report(analysis: dict, english_text: str, risk_score: int, retrieved_context: str = "") -> str:
    if not analysis or "error" in analysis:
        return "Cannot generate report — symptom extraction failed."
    symptoms_text = "\n".join([
        f"  - {s.get('name','?')} | severity: {s.get('severity','?')} | duration: {s.get('duration','?')}"
        for s in analysis.get("symptoms", [])
    ])
    conditions = ", ".join(analysis.get("possible_conditions", [])) or "None identified"
    context_section = f"\nRETRIEVED MEDICAL KNOWLEDGE:\n{retrieved_context}\n" if retrieved_context else ""
    prompt = f"""You are a clinical documentation assistant.
Generate a concise SOAP format medical report based on the data below.
Return ONLY the report — no extra commentary, no markdown headings with #.
PATIENT DATA:
Chief Complaint : {analysis.get('chief_complaint', 'N/A')}
Symptoms       :
{symptoms_text}
Possible Conditions: {conditions}
Urgency        : {analysis.get('urgency', 'N/A')}
Risk Score     : {risk_score}/100
Original Statement: {english_text}
{context_section}
FORMAT TO USE:
SUBJECTIVE:
[what the patient reports]
OBJECTIVE:
[observable findings from the speech/statement]
ASSESSMENT:
[clinical interpretation, possible diagnoses]
PLAN:
[recommended next steps for the treating doctor]
NOTE: This report is AI-generated and must be reviewed by a qualified health professional before any clinical decision is made."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Report generation error: {str(e)}"


# =========================================
#  FULL PIPELINE
# =========================================
def run_full_pipeline(text: str, lang: str) -> dict:
    timings = {}
    t0 = time.time()

    # Translation
    t = time.time()
    english_text = translate_if_needed(text, lang)
    timings["translation_ms"] = round((time.time() - t) * 1000)

    # Classification
    t = time.time()
    category = classify_query(english_text)
    timings["classification_ms"] = round((time.time() - t) * 1000)

    lang_label = "Urdu" if lang == "ur" else ("English" if lang == "en" else "Unknown")

    if category == "crisis":
        return {
            "status": "crisis",
            "english_text": english_text,
            "lang": lang,
            "lang_label": lang_label,
            "timings": timings,
            "total_ms": round((time.time() - t0) * 1000)
        }

    if category == "non_medical":
        return {
            "status": "non_medical",
            "english_text": english_text,
            "lang": lang,
            "lang_label": lang_label,
            "timings": timings,
            "total_ms": round((time.time() - t0) * 1000)
        }

    # Symptom extraction
    t = time.time()
    analysis = extract_symptoms(english_text)
    timings["extraction_ms"] = round((time.time() - t) * 1000)

    # Risk scoring
    t = time.time()
    risk_score = calculate_risk_score(analysis)
    risk_level = get_risk_level(risk_score)
    timings["risk_scoring_ms"] = round((time.time() - t) * 1000)

    # Disease lookup
    t = time.time()
    symptoms_list = analysis.get("symptoms", [])
    dataset_matches = lookup_diseases(symptoms_list, top_n=5)
    timings["disease_lookup_ms"] = round((time.time() - t) * 1000)

    if dataset_matches:
        analysis["possible_conditions"] = [d["disease"] for d in dataset_matches]
        analysis["dataset_match_detail"] = [
            f"{d['disease']} ({d['matched']}/{d['total']} symptoms matched)"
            for d in dataset_matches
        ]
    else:
        analysis["dataset_match_detail"] = ["Dataset lookup returned no matches"]

    # RAG retrieval
    t = time.time()
    rag_query = english_text + " " + " ".join(s.get("name", "") for s in symptoms_list)
    retrieved = retrieve_knowledge(rag_query, n=3)
    timings["rag_retrieval_ms"] = round((time.time() - t) * 1000)

    retrieved_context = "\n\n".join(
        f"[{r['title']}]\n{r['text']}" for r in retrieved
    ) if retrieved else ""

    # SOAP report
    t = time.time()
    soap_report = generate_soap_report(analysis, english_text, risk_score, retrieved_context)
    timings["soap_generation_ms"] = round((time.time() - t) * 1000)

    timings["total_ms"] = round((time.time() - t0) * 1000)

    return {
        "status": "medical",
        "lang": lang,
        "lang_label": lang_label,
        "english_text": english_text,
        "analysis": analysis,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "dataset_matches": dataset_matches,
        "retrieved_chunks": retrieved,
        "soap_report": soap_report,
        "models_used": {
            "transcription": "whisper-large-v3-turbo",
            "classification": "llama-3.1-8b-instant",
            "extraction": "llama-3.3-70b-versatile",
            "soap": "llama-3.3-70b-versatile",
            "embeddings": "all-MiniLM-L6-v2"
        },
        "timings": timings
    }


# =========================================
#  DIRECT PIPELINE ENTRYPOINTS
#  (replaces HTTP API calls)
# =========================================
def call_text_pipeline(text: str) -> dict:
    if not client:
        return {"status": "error", "message": "GROQ_API_KEY not configured"}
    if not text or not text.strip():
        return {"status": "error", "message": "Text cannot be empty"}
    lang = detect_language(text)
    if lang == "other":
        return {"status": "error", "message": "Unsupported language. Only Urdu and English are supported."}
    try:
        return run_full_pipeline(text, lang)
    except Exception as e:
        return {"status": "error", "message": str(e)}


def call_audio_pipeline(audio_bytes: bytes) -> dict:
    if not client:
        return {"status": "error", "message": "GROQ_API_KEY not configured"}
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        t0 = time.time()
        text, lang = transcribe_audio_file(tmp_path)
        transcription_ms = round((time.time() - t0) * 1000)
        os.unlink(tmp_path)

        if lang == "unknown":
            return {"status": "error", "message": text}

        result = run_full_pipeline(text, lang)
        result["original_transcript"] = text
        result["timings"]["transcription_ms"] = transcription_ms
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_system_health() -> dict:
    return {
        "status": "healthy",
        "groq_configured": bool(api_key),
        "rag_ready": _rag_model is not None,
        "disease_db_ready": disease_df is not None,
        "disease_count": len(disease_df) if disease_df is not None else 0,
        "rag_doc_count": len(_rag_docs)
    }


# =========================================
#  STREAMLIT PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="TabeebAI — Clinical Triage",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else "🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

_logo_b64 = ""
_logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
_logo_img = f'<img src="data:image/png;base64,{_logo_b64}" style="width:100%;height:100%;object-fit:contain;border-radius:6px">' if _logo_b64 else "+"

_agahi_b64 = ""
_agahi_path = os.path.join(os.path.dirname(__file__), "assets", "agahi_logo.png")
if os.path.exists(_agahi_path):
    with open(_agahi_path, "rb") as _f:
        _agahi_b64 = base64.b64encode(_f.read()).decode()
_agahi_img = f'<img src="data:image/png;base64,{_agahi_b64}" style="width:100%;height:auto;object-fit:contain">' if _agahi_b64 else "Aagahi Labs"

# =========================================
#  GLOBAL CSS
# =========================================
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=JetBrains+Mono:wght@400;500&family=Noto+Nastaliq+Urdu:wght@400;600;700&display=swap');

@font-face {
    font-family: 'UrduNastaliq';
    src: local('Noto Nastaliq Urdu'),
         url('https://fonts.gstatic.com/s/notonastaliqurdu/v22/LhWNMUPbL95F4oFGQEbPSGJgYtN1g_U.woff2') format('woff2');
    unicode-range: U+0600-06FF, U+0750-077F, U+FB50-FDFF, U+FE70-FEFF;
}

/* ── Design tokens ── */
:root {
    --bg-primary:    #f5f7fa;
    --bg-secondary:  #fafbfd;
    --bg-card:       #ffffff;
    --bg-glass:      rgba(255, 255, 255, 0.9);
    --border-subtle: rgba(15, 23, 42, 0.08);
    --border-accent: rgba(15, 23, 42, 0.18);
    --cyan:          #0d7494;
    --cyan-dim:      rgba(13, 116, 148, 0.10);
    --cyan-glow:     rgba(13, 116, 148, 0.12);
    --red:           #c0392b;
    --red-dim:       rgba(192, 57, 43, 0.08);
    --yellow:        #c97f0a;
    --yellow-dim:    rgba(201, 127, 10, 0.10);
    --green:         #0f7a5a;
    --green-dim:     rgba(15, 122, 90, 0.08);
    --text-primary:  #0f1a2e;
    --text-secondary:#3d4a63;
    --text-muted:    #6b7a91;
    --font-main:     'Inter', 'DM Sans', 'UrduNastaliq', 'Noto Nastaliq Urdu', system-ui, sans-serif;
    --font-mono:     'JetBrains Mono', 'DM Mono', ui-monospace, monospace;
    --font-urdu:     'Noto Nastaliq Urdu', 'UrduNastaliq', serif;
    --radius:        10px;
    --radius-lg:     16px;
    --track:         rgba(15, 23, 42, 0.06);
}

/* ── Audio input widget ── */
[data-testid="stAudioInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stAudioInput"] > div { background: var(--bg-card) !important; }
[data-testid="stAudioInput"] * { background-color: transparent !important; }
[data-testid="stAudioInput"] button {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-accent) !important;
}
[data-testid="stAudioInput"] svg {
    fill: var(--cyan) !important;
    color: var(--cyan) !important;
    stroke: var(--cyan) !important;
}
[data-testid="stAudioInput"] p,
[data-testid="stAudioInput"] span,
[data-testid="stAudioInput"] small,
[data-testid="stAudioInput"] label { color: var(--text-secondary) !important; }

/* ── Audio player ── */
[data-testid="stAudio"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem !important;
}
[data-testid="stAudio"] audio {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    width: 100% !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] *:not(svg):not(path):not(button) {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius) !important;
    box-shadow: none !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
[data-testid="stFileUploader"] button span { color: inherit !important; }
[data-testid="stFileUploader"] svg,
[data-testid="stFileUploader"] path {
    fill: var(--text-secondary) !important;
    stroke: none !important;
}

/* ── Textarea placeholder ── */
.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg-primary) !important; }
.main { background-color: var(--bg-primary) !important; }
.block-container { background-color: transparent !important; }
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
    background: transparent;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1rem 1.25rem 1.5rem 1.25rem !important;
}
[data-testid="stSidebarHeader"] {
    margin-bottom: 0rem !important;
    height: 2.75rem !important;
}
[data-testid="stMainBlockContainer"] {
    padding: 1rem 5rem 5rem !important;
}

/* ── Remove default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }
.stDeployButton { display: none; }

/* ── Radio buttons ── */
[data-testid="stRadio"] label p {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
}
[data-testid="stRadio"] label:hover p { color: var(--text-primary) !important; }
[data-testid="stRadio"] [aria-checked="true"] ~ div p,
[data-testid="stRadio"] input:checked ~ div p {
    color: var(--cyan) !important;
    font-weight: 600 !important;
}

/* ── Buttons — solid accent ── */
.stButton > button,
[data-testid="stDownloadButton"] > button {
    background: var(--cyan) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-main) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    background: #075568 !important;
    box-shadow: none !important;
}
.stButton > button:active,
[data-testid="stDownloadButton"] > button:active { background: #075568 !important; }

/* ── Text inputs ── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s ease !important;
    caret-color: var(--cyan) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px var(--cyan-dim) !important;
    outline: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border-subtle) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: var(--font-main) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    border: none !important;
    transition: color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--cyan) !important;
    box-shadow: none !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ── Metrics — all neutral ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: var(--font-mono) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: var(--cyan) !important;
    border-radius: 999px !important;
}
.stProgress > div > div > div {
    background: var(--track) !important;
    border-radius: 999px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] { background: transparent !important; border: none !important; }
[data-testid="stExpander"] details {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] details summary {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1rem !important;
    list-style: none !important;
}
[data-testid="stExpander"] details[open] summary {
    border-radius: var(--radius) var(--radius) 0 0 !important;
}
[data-testid="stExpander"] details summary:hover,
[data-testid="stExpander"] details summary:focus {
    background: var(--bg-secondary) !important;
    color: var(--cyan) !important;
}
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] details > div > div {
    background: var(--bg-secondary) !important;
    border-top: 1px solid var(--border-subtle) !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ── Tabs content ── */
[data-testid="stTabsContent"] { color: var(--text-primary) !important; }
[data-testid="stTabsContent"] p,
[data-testid="stTabsContent"] span,
[data-testid="stTabsContent"] div { color: inherit; }

/* ── Code / monospace ── */
code, pre, .stCode {
    background: var(--bg-secondary) !important;
    color: var(--cyan) !important;
    font-family: var(--font-mono) !important;
    border-radius: 6px !important;
}
.stCode > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s ease;
}
.card:hover { border-color: var(--border-accent); }
.card-glass {
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 0.5rem 0;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2em 0.75em;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-red    { background: var(--red-dim);    color: var(--red);    border: 1px solid var(--red); }
.badge-yellow { background: var(--yellow-dim); color: var(--yellow); border: 1px solid var(--yellow); }
.badge-green  { background: var(--green-dim);  color: var(--green);  border: 1px solid var(--green); }
.badge-cyan   { background: var(--cyan-dim);   color: var(--cyan);   border: 1px solid var(--cyan); }

/* ── Section title ── */
.section-title {
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* ── Risk bar ── */
.risk-bar-container {
    background: var(--track);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}

/* ── SOAP block ── */
.soap-block {
    background: var(--bg-secondary);
    border-left: 2px solid var(--cyan);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    line-height: 1.65;
    white-space: pre-wrap;
    color: var(--text-secondary);
}

/* ── Symptom chip ── */
.symptom-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--cyan-dim);
    border: 1px solid var(--border-accent);
    border-radius: 999px;
    padding: 0.25em 0.85em;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--cyan);
    margin: 0.2rem;
}

/* ── Pipeline step ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: var(--radius);
    border: 1px solid var(--border-subtle);
    background: var(--bg-card);
    margin: 0.3rem 0;
    font-size: 0.85rem;
}
.pipeline-step .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-cyan   { background: var(--cyan); }
.dot-green  { background: var(--green); }
.dot-yellow { background: var(--yellow); }
.dot-red    { background: var(--red); }

/* ── Timing pill ── */
.timing-pill {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    text-align: center;
}
.timing-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    display: block;
    font-family: var(--font-mono);
}
.timing-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── RAG chunk ── */
.rag-chunk {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.84rem;
}
.rag-title {
    color: var(--cyan);
    font-weight: 600;
    font-size: 0.82rem;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.rag-score {
    float: right;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* ── Alert blocks ── */
.alert-emergency {
    background: var(--red-dim);
    border: 1.5px solid var(--red);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
}
.alert-caution {
    background: var(--yellow-dim);
    border: 1.5px solid var(--yellow);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
}
.alert-safe {
    background: var(--green-dim);
    border: 1.5px solid var(--green);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
}
.alert-crisis {
    background: rgba(124, 58, 237, 0.07);
    border: 1.5px solid #7c3aed;
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
}
.alert-info {
    background: var(--cyan-dim);
    border: 1.5px solid var(--cyan);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
}
.alert-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.6rem; letter-spacing: 0.01em; }
.alert-body  { font-size: 0.875rem; line-height: 1.65; color: var(--text-primary); }
.alert-list  { margin: 0.6rem 0 0 1.1rem; padding: 0; font-size: 0.875rem; line-height: 1.75; color: var(--text-secondary); }

/* ── Disclaimer bar ── */
.disclaimer-bar {
    background: rgba(201, 127, 10, 0.06);
    border: 1px solid rgba(201, 127, 10, 0.20);
    border-radius: var(--radius);
    padding: 0.9rem 1.25rem;
    font-size: 0.82rem;
    line-height: 1.6;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}
.disclaimer-bar strong { color: var(--yellow); }
.disclaimer-urdu {
    margin-top: 0.5rem;
    direction: rtl;
    text-align: right;
    font-family: var(--font-urdu);
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* ── Header bar ── */
.header-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}
.header-logo {
    width: 64px; height: 64px;
    background: var(--cyan);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.header-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    line-height: 1.15;
}
.header-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 400;
    letter-spacing: 0.02em;
}
.header-status {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--text-muted);
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.45; }
}

.urgency-meter {
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    height: 10px;
    background: var(--track);
}

.sidebar-nav-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0.4rem 0 0.2rem 0;
}
.sidebar-sep {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1em 0px;
}
hr.sidebar-sep {
    margin: 1em 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Ctrl+Enter in the symptom textarea → click Analyze
st.markdown("""
<script>
(function() {
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const active = document.activeElement;
            if (!active || active.tagName !== 'TEXTAREA') return;
            const btns = Array.from(document.querySelectorAll('button'));
            const analyzeBtn = btns.find(b => b.innerText.trim() === 'Analyze');
            if (analyzeBtn && !analyzeBtn.disabled) {
                e.preventDefault();
                analyzeBtn.click();
            }
        }
    });
})();
</script>
""", unsafe_allow_html=True)


# =========================================
#  SESSION STATE
# =========================================
if "result" not in st.session_state:
    st.session_state.result = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"
# HITL review state
if "soap_source" not in st.session_state:
    st.session_state.soap_source = None      # fingerprint to detect new reports
if "soap_edited" not in st.session_state:
    st.session_state.soap_edited = None      # doctor-edited SOAP text
if "soap_confirmed" not in st.session_state:
    st.session_state.soap_confirmed = False
if "soap_reviewer_name" not in st.session_state:
    st.session_state.soap_reviewer_name = ""
if "soap_reviewer_role" not in st.session_state:
    st.session_state.soap_reviewer_role = ""
if "soap_confirmed_at" not in st.session_state:
    st.session_state.soap_confirmed_at = None
# Accessibility
FONT_SCALES = [0.85, 1.0, 1.15, 1.3, 1.5]
FONT_LABELS = ["Small", "Normal", "Large", "X-Large", "XX-Large"]
if "font_idx" not in st.session_state:
    st.session_state.font_idx = 2  # default: Large


# =========================================
#  COMPONENT HELPERS
# =========================================
def render_risk_bar(score: int, risk_level: str):
    if risk_level == "RED":
        color, label = "#c0392b", "Emergency"
    elif risk_level == "YELLOW":
        color, label = "#c97f0a", "Moderate"
    else:
        color, label = "#0f7a5a", "Low Risk"

    pct = score
    st.markdown(f"""
    <div style="margin:0.25rem 0 0.5rem 0">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem">
        <span style="font-size:0.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Risk Level</span>
        <span style="font-size:0.75rem;font-weight:700;color:{color}">{label}</span>
      </div>
      <div style="background:var(--bg-secondary);border-radius:999px;height:8px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;transition:width 0.4s ease"></div>
      </div>
      <div style="text-align:right;font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);margin-top:0.3rem">{score}/100</div>
    </div>
    """, unsafe_allow_html=True)


def render_alert(status: str, result: dict):
    score = result.get("risk_score", 0)
    level = result.get("risk_level", "GREEN")
    analysis = result.get("analysis", {})
    complaint = analysis.get("chief_complaint", "N/A") if analysis else "N/A"
    conditions = ", ".join(analysis.get("possible_conditions", [])) if analysis else "N/A"

    if status == "crisis":
        st.markdown("""
        <div class="alert-crisis">
          <div class="alert-title" style="color:#7c3aed">You Are Not Alone</div>
          <div class="alert-body">
            It sounds like you or someone around you may be in emotional distress.
            TabeebAI cannot provide mental health support, but <strong>trained counsellors are available right now</strong>.
          </div>
          <ul class="alert-list">
            <li><strong>Umang</strong> — 0317-4288665 (24/7)</li>
            <li><strong>Rozan Counselling</strong> — 051-2890505</li>
            <li><strong>Rescue</strong> — 1122</li>
          </ul>
          <div style="font-size:0.8rem;color:#7c3aed;opacity:0.75;margin-top:0.8rem;font-style:italic;direction:rtl;text-align:right;font-family:var(--font-urdu)">
            یہ ایپ ذہنی صحت کی مدد کے لیے نہیں ہے۔ براہ کرم اوپر دیے گئے نمبروں پر کال کریں۔
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if status == "non_medical":
        st.markdown("""
        <div class="alert-info">
          <div class="alert-title" style="color:var(--cyan)">Non-Medical Input Detected</div>
          <div class="alert-body">
            TabeebAI is designed to assist with <strong>patient symptom triage</strong> only.
            Please describe the patient's medical symptoms or health concerns.
          </div>
          <div style="font-size:0.9rem;color:var(--text-muted);margin-top:0.8rem;
                      font-family:var(--font-urdu);direction:rtl;line-height:2.2">
            براہ کرم مریض کی علامات یا صحت کے مسائل بیان کریں۔
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if level == "RED":
        st.markdown(f"""
        <div class="alert-emergency">
          <div class="alert-title" style="color:var(--red)">EMERGENCY ALERT — Immediate Action Required</div>
          <div class="alert-body">
            <strong>Chief Complaint:</strong> {complaint}<br>
            <strong>Risk Score:</strong> {score}/100 &nbsp;&nbsp;
            <strong>Possible Conditions:</strong> {conditions}
          </div>
          <ul class="alert-list">
            <li>Call emergency services immediately — Rescue 1122 / Edhi 115</li>
            <li>Do not leave the patient alone</li>
            <li>Keep patient calm and still</li>
            <li>Prepare for immediate hospital transfer</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    elif level == "YELLOW":
        st.markdown(f"""
        <div class="alert-caution">
          <div class="alert-title" style="color:var(--yellow)">CAUTION — Medical Attention Recommended</div>
          <div class="alert-body">
            <strong>Chief Complaint:</strong> {complaint}<br>
            <strong>Risk Score:</strong> {score}/100 &nbsp;&nbsp;
            <strong>Possible Conditions:</strong> {conditions}
          </div>
          <ul class="alert-list">
            <li>Schedule a doctor's appointment within 24 hours</li>
            <li>Monitor symptoms closely for worsening</li>
            <li>Seek urgent care if new symptoms develop</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe">
          <div class="alert-title" style="color:var(--green)">LOW RISK — Home Care Advised</div>
          <div class="alert-body">
            <strong>Chief Complaint:</strong> {complaint}<br>
            <strong>Risk Score:</strong> {score}/100 &nbsp;&nbsp;
            <strong>Possible Conditions:</strong> {conditions}
          </div>
          <ul class="alert-list">
            <li>Rest and home care</li>
            <li>Stay hydrated and monitor temperature</li>
            <li>Visit a clinic if symptoms persist beyond 3 days</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)


def render_symptoms_chips(symptoms: list):
    _chip_styles = {
        "severe":   {"color": "#c0392b", "bg": "rgba(192,57,43,0.08)",   "border": "rgba(192,57,43,0.30)"},
        "moderate": {"color": "#c97f0a", "bg": "rgba(201,127,10,0.10)",  "border": "rgba(201,127,10,0.30)"},
    }
    _chip_default = {"color": "#0d7494", "bg": "rgba(13,116,148,0.10)", "border": "rgba(13,116,148,0.25)"}
    chips_html = ""
    for s in symptoms:
        name = s.get("name", "")
        sev  = s.get("severity", "mild").lower()
        dur  = s.get("duration", "")
        cs   = _chip_styles.get(sev, _chip_default)
        chips_html += f"""<span style="display:inline-flex;align-items:center;gap:0.35rem;background:{cs['bg']};border:1px solid {cs['border']};border-radius:999px;padding:0.25em 0.9em;font-size:0.78rem;font-weight:500;color:{cs['color']};margin:0.2rem">{name}<span style="opacity:0.35;font-size:0.72rem">|</span><span style="font-size:0.72rem;opacity:0.85">{sev}</span></span>"""
    st.markdown(f'<div style="line-height:2.2">{chips_html}</div>', unsafe_allow_html=True)


def render_soap_report(soap: str):
    sections = {"SUBJECTIVE:": [], "OBJECTIVE:": [], "ASSESSMENT:": [], "PLAN:": []}
    current = None
    for line in soap.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SUBJECTIVE"):
            current = "SUBJECTIVE:"
        elif stripped.upper().startswith("OBJECTIVE"):
            current = "OBJECTIVE:"
        elif stripped.upper().startswith("ASSESSMENT"):
            current = "ASSESSMENT:"
        elif stripped.upper().startswith("PLAN"):
            current = "PLAN:"
        elif current:
            sections[current].append(line)

    icons = {"SUBJECTIVE:": "S", "OBJECTIVE:": "O", "ASSESSMENT:": "A", "PLAN:": "P"}
    colors = {"SUBJECTIVE:": "#0d7494", "OBJECTIVE:": "#0f7a5a", "ASSESSMENT:": "#c97f0a", "PLAN:": "#7c3aed"}
    color_bg = {"SUBJECTIVE:": "rgba(13,116,148,0.10)", "OBJECTIVE:": "rgba(15,122,90,0.08)", "ASSESSMENT:": "rgba(201,127,10,0.10)", "PLAN:": "rgba(124,58,237,0.08)"}
    color_border = {"SUBJECTIVE:": "rgba(13,116,148,0.25)", "OBJECTIVE:": "rgba(15,122,90,0.20)", "ASSESSMENT:": "rgba(201,127,10,0.25)", "PLAN:": "rgba(124,58,237,0.20)"}

    for key, lines in sections.items():
        content = "\n".join(l for l in lines if l.strip())
        if not content:
            continue
        c  = colors[key]
        bg = color_bg[key]
        bd = color_border[key]
        st.markdown(f"""
        <div style="display:flex;gap:0.9rem;margin:0.6rem 0">
          <div style="width:28px;height:28px;border-radius:6px;background:{bg};border:1px solid {bd};display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:800;color:{c};flex-shrink:0;margin-top:2px">{icons[key]}</div>
          <div style="flex:1">
            <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{c};margin-bottom:0.35rem">{key.rstrip(':')}</div>
            <div style="background:var(--bg-secondary);border-left:2px solid {bd};border-radius:0 8px 8px 0;padding:0.75rem 1rem;font-size:0.85rem;line-height:1.7;color:var(--text-secondary);white-space:pre-wrap">{content}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================
#  SIDEBAR
# =========================================
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:0.5rem">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.3rem">
        <div style="width:36px;height:36px;border-radius:9px;overflow:hidden;flex-shrink:0;display:flex;align-items:center;justify-content:center">{_agahi_img}</div>
        <div>
          <div style="font-size:1.05rem;font-weight:700;letter-spacing:-0.01em;color:var(--text-primary)">Aagahi Labs</div>
          <div style="font-size:0.72rem;color:var(--text-secondary);letter-spacing:0.04em">Presents</div>
        </div>
      </div>
    </div>
    <hr class="sidebar-sep">
    """, unsafe_allow_html=True)

    health = get_system_health()
    if health:
        st.markdown(f"""
        <div style="background:var(--green-dim);border:1px solid var(--green);border-radius:var(--radius);padding:0.6rem 0.9rem;margin-bottom:1rem">
          <div style="display:flex;align-items:center;gap:0.5rem">
            <div class="status-dot"></div>
            <span style="font-size:0.78rem;font-weight:600;color:var(--green)">System Online</span>
          </div>
          <div style="font-size:0.74rem;color:var(--text-muted);margin-top:0.3rem;font-family:var(--font-mono)">
            {health.get('disease_count', 0)} diseases &middot; {health.get('rag_doc_count', 0)} RAG docs
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)

    dashboard = st.radio(
        "Select View",
        ["Patient Dashboard", "Doctor Dashboard", "Developer Dashboard"],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="sidebar-sep">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-nav-label">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input",
        ["Text Input", "Audio Input"],
        label_visibility="collapsed",
        key="input_mode_radio"
    )

    st.markdown('<hr class="sidebar-sep">', unsafe_allow_html=True)
    with st.expander("Accessibility", expanded=False):
        _fi = st.session_state.font_idx
        _fa_col, _fl_col, _fp_col = st.columns([1, 2, 1])
        with _fa_col:
            if st.button("A−", key="font_dec", disabled=(_fi == 0)):
                st.session_state.font_idx -= 1
                st.rerun()
        with _fl_col:
            st.markdown(
                f'<div style="text-align:center;font-size:0.78rem;color:var(--text-secondary);'
                f'font-weight:600;padding:0.45rem 0">{FONT_LABELS[_fi]}</div>',
                unsafe_allow_html=True,
            )
        with _fp_col:
            if st.button("A+", key="font_inc", disabled=(_fi == len(FONT_SCALES) - 1)):
                st.session_state.font_idx += 1
                st.rerun()

    st.markdown('<hr class="sidebar-sep">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:var(--text-secondary);line-height:1.6">
      <strong style="color:var(--yellow)">Disclaimer</strong><br>
      TabeebAI is a clinical decision support tool. Not a substitute for professional medical judgement.
      All outputs must be reviewed by a licensed clinician.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.result:
        st.markdown('<hr class="sidebar-sep">', unsafe_allow_html=True)
        if st.button("Clear Results", use_container_width=True):
            st.session_state.result = None
            st.rerun()


# =========================================
#  MAIN CONTENT AREA
# =========================================

# Dynamic font scale — injected fresh every render so sidebar controls take effect immediately
_font_px = round(16 * FONT_SCALES[st.session_state.font_idx], 2)
st.markdown(
    f"<style>html {{ font-size: {_font_px}px !important; }}</style>",
    unsafe_allow_html=True,
)

# Header
st.markdown(f"""
<div class="header-bar">
  <div class="header-logo" style="padding:4px">{_logo_img}</div>
  <div>
    <div class="header-title">TabeebAI</div>
    <div class="header-subtitle">AI-Powered Clinical Triage — Urdu / English</div>
  </div>
  <div class="header-status">
    <div class="status-dot"></div>
    System Active
  </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-bar">
  <strong>Important:</strong> TabeebAI is a <strong>clinical decision support tool</strong> intended to assist qualified healthcare professionals.
  It is <strong>not a diagnostic tool</strong> and must not replace professional medical judgement. All AI-generated reports must be reviewed by a licensed medical professional before any action is taken.
  <div class="disclaimer-urdu">یہ ایپ صرف ڈاکٹروں کی مدد کے لیے ہے۔ کوئی بھی طبی فیصلہ کرنے سے پہلے ڈاکٹر سے مشورہ کریں۔</div>
</div>
""", unsafe_allow_html=True)


# =========================================
#  INPUT SECTION
# =========================================
def render_input_section():
    st.markdown('<div class="section-title">Patient Input</div>', unsafe_allow_html=True)

    if input_mode == "Text Input":
        col_in, col_btn = st.columns([5, 1])

        with col_in:
            user_text = st.text_area(
                "Enter symptoms in Urdu or English",
                placeholder=(
                    "Example: I have severe chest pain and difficulty breathing since morning...\n"
                    "مثال: مجھے سینے میں درد ہے اور سانس لینے میں دشواری ہو رہی ہے..."
                ),
                height=120,
                label_visibility="collapsed",
                key="text_input"
            )

        with col_btn:
            st.markdown("""
            <div style="font-size:0.68rem;color:var(--text-muted);text-align:center;
                        margin-bottom:0.3rem;letter-spacing:0.03em">
              Ctrl+Enter
            </div>
            """, unsafe_allow_html=True)
            analyze_clicked = st.button("Analyze", use_container_width=True, type="primary")

        if analyze_clicked:
            if not user_text or not user_text.strip():
                st.warning("Please enter some text before analyzing.")
                return

            with st.spinner("Running clinical pipeline..."):
                result = call_text_pipeline(user_text)
                result["original_input"] = user_text

            st.session_state.result = result
            st.rerun()

    # =========================
    # AUDIO INPUT MODE
    # =========================
    else:
        st.markdown("""
        <div style="background:rgba(201,127,10,0.06);border:1px solid rgba(201,127,10,0.20);
                    border-radius:var(--radius);padding:0.75rem 1rem;margin-bottom:1rem;
                    font-size:0.82rem;color:#c97f0a">
          <strong>Microphone not working?</strong>
          Browsers block mic access inside embedded frames.
          <a href="?" target="_blank"
             style="color:#38bdf8;text-decoration:underline;margin-left:0.3rem">
            Open the app in a new tab 
          </a>
          to enable live recording, or use <strong>Upload File</strong> below.
        </div>
        """, unsafe_allow_html=True)

        col_rec, col_up = st.columns(2)

        audio_bytes = None
        source_label = None

        # ── COLUMN 1: Live microphone recording ──
        with col_rec:
            st.markdown(
                '<div style="font-size:0.78rem;color:var(--text-muted);text-transform:uppercase;'
                'letter-spacing:0.06em;margin-bottom:0.5rem">Live Recording</div>',
                unsafe_allow_html=True
            )
            audio_value = st.audio_input(
                "Record patient voice",
                label_visibility="collapsed",
                key="live_recorder"
            )
            if audio_value is not None:
                audio_bytes = audio_value.read()
                source_label = "live recording"

        # ── COLUMN 2: File upload fallback ──
        with col_up:
            st.markdown(
                '<div style="font-size:0.78rem;color:var(--text-muted);text-transform:uppercase;'
                'letter-spacing:0.06em;margin-bottom:0.5rem">Upload File</div>',
                unsafe_allow_html=True
            )
            uploaded = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                label_visibility="collapsed",
                key="audio_upload"
            )
            if uploaded is not None and audio_bytes is None:
                audio_bytes = uploaded.read()
                source_label = f"uploaded — {uploaded.name}"

        # ── Playback + Analyze button ──
        if audio_bytes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/wav")
            st.markdown(
                f'<div style="font-size:0.74rem;color:var(--text-muted);margin:0.3rem 0 0.7rem 0">'
                f'Source: {source_label}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div style="background:var(--bg-card);border:1px dashed var(--border-accent);
                        border-radius:var(--radius);padding:1.25rem;text-align:center;
                        color:var(--text-muted);font-size:0.82rem;margin-top:0.5rem">
              Record using the microphone above, or upload an audio file
            </div>
            """, unsafe_allow_html=True)

        # ── Submit button always visible ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "Transcribe & Analyze",
            use_container_width=True,
            type="primary",
            key="audio_analyze_btn",
            disabled=(audio_bytes is None)
        ):
            if audio_bytes is None:
                st.warning("Please record or upload audio first.")
            else:
                with st.spinner("Transcribing audio and running clinical pipeline..."):
                    result = call_audio_pipeline(audio_bytes)
                st.session_state.result = result
                st.rerun()


render_input_section()

# =========================================
#  NO RESULT STATE
# =========================================
if st.session_state.result is None:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 2rem;color:var(--text-muted)">
      <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4">+</div>
      <div style="font-size:1rem;font-weight:500;margin-bottom:0.4rem;color:var(--text-secondary)">No Analysis Yet</div>
      <div style="font-size:0.82rem">Enter patient symptoms above to begin clinical triage</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================================
#  RESULTS
# =========================================
result = st.session_state.result
status = result.get("status", "error")
analysis = result.get("analysis", {}) or {}
risk_score = result.get("risk_score", 0)
risk_level = result.get("risk_level", "GREEN")
soap = result.get("soap_report", "")
retrieved = result.get("retrieved_chunks", [])
timings = result.get("timings", {})
lang_label = result.get("lang_label", "Unknown")
english_text = result.get("english_text", "")
original_input = result.get("original_input", result.get("original_transcript", ""))
dataset_matches = result.get("dataset_matches", [])

st.markdown("<br>", unsafe_allow_html=True)

# ── Alert banner always shown ──────────────────────────────
render_alert(status, result)

if status in ("crisis", "non_medical", "error"):
    if status == "error":
        st.error(f"Pipeline Error: {result.get('message', 'Unknown error')}")
    st.stop()

# =========================================
#  DASHBOARD TABS
# =========================================
if dashboard == "Patient Dashboard":

    st.markdown('<div class="section-title">Patient Overview</div>', unsafe_allow_html=True)

    # Key metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Risk Score", f"{risk_score}/100")
    with m2:
        lvl_map = {"RED": "Emergency", "YELLOW": "Moderate", "GREEN": "Low Risk"}
        st.metric("Urgency", lvl_map.get(risk_level, risk_level))
    with m3:
        st.metric("Language", lang_label)
    with m4:
        n_sym = len(analysis.get("symptoms", []))
        st.metric("Symptoms Found", str(n_sym))

    st.markdown("<br>", unsafe_allow_html=True)

    # Urgency meter
    render_risk_bar(risk_score, risk_level)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column layout
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Reported Symptoms</div>', unsafe_allow_html=True)
        symptoms = analysis.get("symptoms", [])
        if symptoms:
            render_symptoms_chips(symptoms)
        else:
            st.markdown('<div style="color:var(--text-muted);font-size:0.85rem">No specific symptoms extracted.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Chief Complaint</div>', unsafe_allow_html=True)
        complaint = analysis.get("chief_complaint", "Not specified")
        st.markdown(f"""
        <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:1rem;font-size:0.88rem;color:var(--text-secondary);line-height:1.6">
          {complaint}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">What You Should Do Next</div>', unsafe_allow_html=True)

        if risk_level == "RED":
            steps = [
                ("Call 1122 / 115 immediately", "red"),
                ("Do not leave the patient alone", "red"),
                ("Keep patient calm and still", "yellow"),
                ("Prepare for hospital transfer", "yellow"),
            ]
        elif risk_level == "YELLOW":
            steps = [
                ("See a doctor within 24 hours", "yellow"),
                ("Monitor symptoms for worsening", "yellow"),
                ("Seek urgent care if new symptoms develop", "cyan"),
                ("Keep track of symptom duration", "cyan"),
            ]
        else:
            steps = [
                ("Rest and home care is appropriate", "green"),
                ("Stay hydrated and monitor temperature", "green"),
                ("Visit a clinic if symptoms persist > 3 days", "cyan"),
                ("Avoid strenuous activity until recovered", "cyan"),
            ]

        for step, color in steps:
            dot_class = f"dot-{color}"
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="dot {dot_class}"></div>
              <span style="font-size:0.85rem;color:var(--text-secondary)">{step}</span>
            </div>
            """, unsafe_allow_html=True)

        if original_input and lang_label == "Urdu":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Translation</div>', unsafe_allow_html=True)
            col_ur, col_en = st.columns(2)
            with col_ur:
                st.markdown(f"""
                <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:0.75rem;font-size:0.82rem;color:var(--text-secondary)">
                  <div style="font-size:0.72rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.35rem">Original (Urdu)</div>
                  {original_input}
                </div>
                """, unsafe_allow_html=True)
            with col_en:
                st.markdown(f"""
                <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:0.75rem;font-size:0.82rem;color:var(--text-secondary)">
                  <div style="font-size:0.72rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.35rem">English</div>
                  {english_text}
                </div>
                """, unsafe_allow_html=True)


elif dashboard == "Doctor Dashboard":

    st.markdown('<div class="section-title">Clinical Overview</div>', unsafe_allow_html=True)

    # Clinical metrics
    cm1, cm2, cm3, cm4 = st.columns(4)
    with cm1:
        st.metric("Risk Score", f"{risk_score}/100")
    with cm2:
        urgency = analysis.get("urgency", "N/A").upper()
        st.metric("Triage Urgency", urgency)
    with cm3:
        st.metric("Input Language", lang_label)
    with cm4:
        st.metric("Symptoms Extracted", len(analysis.get("symptoms", [])))

    st.markdown("<br>", unsafe_allow_html=True)

    render_risk_bar(risk_score, risk_level)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main clinical area
    tab_soap, tab_symptoms, tab_diseases, tab_rag = st.tabs([
        "SOAP Report", "Symptom Analysis", "Disease Matching", "Retrieved Knowledge"
    ])

    with tab_soap:
        if soap:
            # Reset HITL state whenever a brand-new report arrives
            if st.session_state.soap_source != soap:
                st.session_state.soap_source    = soap
                st.session_state.soap_edited    = soap
                st.session_state.soap_confirmed = False
                st.session_state.soap_reviewer_name = ""
                st.session_state.soap_reviewer_role = ""
                st.session_state.soap_confirmed_at  = None

            # ── Confirmation banner ────────────────────────────────────
            if st.session_state.soap_confirmed:
                st.markdown(f"""
                <div style="background:rgba(52,211,153,0.10);border:1.5px solid #34d399;
                            border-radius:var(--radius-lg);padding:1rem 1.4rem;
                            margin-bottom:1.25rem;display:flex;align-items:center;gap:0.9rem">
                  <div style="font-size:1.5rem;line-height:1">✅</div>
                  <div>
                    <div style="font-weight:700;color:#34d399;font-size:0.92rem;
                                letter-spacing:0.01em">Report Confirmed</div>
                    <div style="font-size:0.82rem;color:var(--text-secondary);margin-top:0.2rem">
                      Reviewed by
                      <strong style="color:var(--text-primary)">
                        {st.session_state.soap_reviewer_name}
                      </strong>
                      &nbsp;·&nbsp;
                      <span style="color:var(--text-muted)">
                        {st.session_state.soap_reviewer_role}
                      </span>
                      &nbsp;·&nbsp;
                      <span style="font-family:var(--font-mono);font-size:0.79rem">
                        {st.session_state.soap_confirmed_at}
                      </span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Attractive rendered SOAP view (always shown) ──────────
            section_label = (
                "SOAP Report — confirmed" if st.session_state.soap_confirmed
                else "SOAP Report"
            )
            st.markdown(f'<div class="section-title">{section_label}</div>',
                        unsafe_allow_html=True)
            render_soap_report(st.session_state.soap_edited or soap)

            # ── Editable textarea in expander (only when not confirmed) ──
            if not st.session_state.soap_confirmed:
                with st.expander("✏️  Edit Report", expanded=False):
                    edited_soap = st.text_area(
                        "SOAP Report",
                        value=st.session_state.soap_edited,
                        height=300,
                        label_visibility="collapsed",
                        key="soap_edit_area",
                    )
                    st.session_state.soap_edited = edited_soap

            # ── Doctor Review row ──────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem">
              <div style="width:3px;height:18px;background:var(--cyan);border-radius:2px"></div>
              <span class="section-title" style="margin:0;border:none;padding:0">Doctor Review</span>
            </div>
            """, unsafe_allow_html=True)

            rev_c1, rev_c2, rev_c3 = st.columns([2, 2, 1])
            if st.session_state.soap_confirmed:
                # Show confirmed values as readable text, not disabled inputs
                with rev_c1:
                    st.markdown(f"""
                    <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.3rem">Doctor Name</div>
                    <div style="font-size:0.92rem;font-weight:600;
                                color:var(--text-primary)">{st.session_state.soap_reviewer_name}</div>
                    """, unsafe_allow_html=True)
                with rev_c2:
                    st.markdown(f"""
                    <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.3rem">Role / Specialisation</div>
                    <div style="font-size:0.92rem;color:var(--text-secondary)">
                      {st.session_state.soap_reviewer_role}</div>
                    """, unsafe_allow_html=True)
                reviewer_name = st.session_state.soap_reviewer_name
                reviewer_role = st.session_state.soap_reviewer_role
            else:
                with rev_c1:
                    reviewer_name = st.text_input(
                        "Doctor Name",
                        placeholder="Dr. Ahmed Khan",
                        key="reviewer_name_input",
                    )
                with rev_c2:
                    reviewer_role = st.text_input(
                        "Role / Specialisation",
                        placeholder="e.g. General Practitioner",
                        key="reviewer_role_input",
                    )
            with rev_c3:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.soap_confirmed:
                    if st.button("✔ Confirm Report", type="primary",
                                 use_container_width=True, key="confirm_btn"):
                        if not reviewer_name.strip():
                            st.warning("Please enter the reviewing doctor's name.")
                        else:
                            st.session_state.soap_confirmed    = True
                            st.session_state.soap_reviewer_name = reviewer_name.strip()
                            st.session_state.soap_reviewer_role = (
                                reviewer_role.strip() or "Physician"
                            )
                            st.session_state.soap_confirmed_at = time.strftime(
                                "%Y-%m-%d  %H:%M:%S"
                            )
                            st.rerun()
                else:
                    if st.button("↩ Reopen for Edit", use_container_width=True,
                                 key="reopen_btn"):
                        st.session_state.soap_confirmed = False
                        st.rerun()

            # ── Download buttons (use doctor-edited text) ──────────────
            final_soap = st.session_state.soap_edited or soap
            reviewer_line = (
                f"\n\nReviewed by: {st.session_state.soap_reviewer_name}"
                f" ({st.session_state.soap_reviewer_role})"
                f"  —  {st.session_state.soap_confirmed_at}"
                if st.session_state.soap_confirmed else
                "\n\n[Pending doctor review]"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            col_dl1, col_dl2, col_space = st.columns([1, 1, 3])
            with col_dl1:
                st.download_button(
                    "Download Report (.txt)",
                    data=final_soap + reviewer_line,
                    file_name="tabeebai_soap_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_dl2:
                md_report = f"""# TabeebAI Clinical Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Risk Score:** {risk_score}/100
**Risk Level:** {risk_level}
**Language:** {lang_label}

---

## Chief Complaint
{analysis.get("chief_complaint", "N/A")}

---

{final_soap}

---
{reviewer_line}

*AI-generated report — must be reviewed by a licensed medical professional.*
"""
                st.download_button(
                    "Download Report (.md)",
                    data=md_report,
                    file_name="tabeebai_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        else:
            st.info("No SOAP report generated.")

    with tab_symptoms:
        symptoms = analysis.get("symptoms", [])
        if symptoms:
            st.markdown('<div class="section-title">Extracted Symptoms</div>', unsafe_allow_html=True)
            render_symptoms_chips(symptoms)
            st.markdown("<br>", unsafe_allow_html=True)

            for s in symptoms:
                sev = s.get("severity", "mild").lower()
                _sev_map = {
                    "severe":   {"color": "#c0392b", "bg": "rgba(192,57,43,0.08)",  "border": "rgba(192,57,43,0.30)"},
                    "moderate": {"color": "#c97f0a", "bg": "rgba(201,127,10,0.10)", "border": "rgba(201,127,10,0.30)"},
                    "mild":     {"color": "#0f7a5a", "bg": "rgba(15,122,90,0.08)",  "border": "rgba(15,122,90,0.25)"},
                }
                sc = _sev_map.get(sev, {"color": "#0d7494", "bg": "rgba(13,116,148,0.10)", "border": "rgba(13,116,148,0.25)"})
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;background:var(--bg-secondary);border:1px solid var(--border-subtle);border-radius:var(--radius);padding:0.75rem 1rem;margin:0.35rem 0">
                  <div style="font-weight:500;font-size:0.87rem;color:var(--text-primary)">{s.get("name", "—")}</div>
                  <div style="display:flex;gap:1rem;align-items:center">
                    <span style="font-size:0.75rem;color:var(--text-secondary)">Duration: <strong style="color:var(--text-primary)">{s.get("duration", "N/A")}</strong></span>
                    <span style="background:{sc['bg']};border:1px solid {sc['border']};border-radius:999px;padding:0.15em 0.7em;font-size:0.72rem;font-weight:600;color:{sc['color']};text-transform:uppercase">{sev}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No symptoms extracted.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transcript</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"""
            <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:1rem;font-size:0.85rem;color:var(--text-secondary);line-height:1.65">
              <div style="font-size:0.72rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem">Original Input</div>
              {original_input or "—"}
            </div>
            """, unsafe_allow_html=True)
        with t2:
            st.markdown(f"""
            <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:1rem;font-size:0.85rem;color:var(--text-secondary);line-height:1.65">
              <div style="font-size:0.72rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem">English Translation</div>
              {english_text or "—"}
            </div>
            """, unsafe_allow_html=True)

    with tab_diseases:
        st.markdown('<div class="section-title">Dataset-Matched Diseases</div>', unsafe_allow_html=True)
        if dataset_matches:
            max_matched = max(d.get("matched", 0) for d in dataset_matches) or 1
            for d in dataset_matches:
                matched = d.get("matched", 0)
                total = d.get("total", 1)
                pct = int((matched / max_matched) * 100)
                bar_color = "#c0392b" if pct > 70 else "#c97f0a" if pct > 40 else "#0d7494"
                st.markdown(f"""
                <div style="background:var(--bg-secondary);border:1px solid var(--border-subtle);border-radius:var(--radius);padding:0.9rem 1.1rem;margin:0.35rem 0">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                    <span style="font-weight:600;font-size:0.88rem;color:var(--text-primary)">{d.get("disease","—")}</span>
                    <span style="font-family:var(--font-mono);font-size:0.78rem;color:var(--text-secondary)">{matched}/{total} symptoms</span>
                  </div>
                  <div style="background:var(--bg-card);border-radius:999px;height:6px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:999px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No disease matches found. Ensure diseases_symptoms.csv is in the data/ folder.")

        detail = analysis.get("dataset_match_detail", [])
        if detail:
            with st.expander("Match Detail"):
                for item in detail:
                    st.markdown(f'<div style="font-size:0.82rem;color:var(--text-secondary);padding:0.2rem 0">{item}</div>', unsafe_allow_html=True)

    with tab_rag:
        st.markdown('<div class="section-title">Retrieved Medical Knowledge</div>', unsafe_allow_html=True)
        if retrieved:
            for chunk in retrieved:
                score_pct = int(chunk.get("score", 0) * 100)
                score_color = "#0f7a5a" if score_pct > 70 else "#c97f0a" if score_pct > 50 else "#6b7a91"
                with st.expander(f"{chunk.get('title', 'Document')}   —   Relevance: {score_pct}%"):
                    st.markdown(f"""
                    <div style="font-size:0.85rem;color:var(--text-primary);line-height:1.7;padding:0.5rem 0">
                      {chunk.get("text", "")}
                    </div>
                    <div style="margin-top:0.75rem">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem">
                        <span style="font-size:0.72rem;color:var(--text-secondary)">Cosine Similarity</span>
                        <span style="font-family:var(--font-mono);font-size:0.72rem;color:{score_color}">{chunk.get('score',0):.4f}</span>
                      </div>
                      <div style="background:var(--bg-card);border-radius:999px;height:4px">
                        <div style="width:{score_pct}%;height:100%;background:{score_color};border-radius:999px"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No RAG chunks retrieved. Ensure medical_knowledge.json is in the data/ folder.")


elif dashboard == "Developer Dashboard":

    st.markdown('<div class="section-title">Pipeline Observability</div>', unsafe_allow_html=True)

    dev_tab1, dev_tab2, dev_tab3, dev_tab4, dev_tab5 = st.tabs([
        "Pipeline Status", "Timings", "Raw JSON", "RAG Debug", "Models"
    ])

    with dev_tab1:
        st.markdown('<div class="section-title">Execution Pipeline</div>', unsafe_allow_html=True)

        pipeline_steps = [
            ("Language Detection", lang_label, "green"),
            ("Query Classification", "medical", "green"),
            ("Translation", "Urdu → English" if lang_label == "Urdu" else "Skipped (English)", "cyan" if lang_label == "Urdu" else "yellow"),
            ("Symptom Extraction", f"{len(analysis.get('symptoms',[]))} symptoms extracted", "green"),
            ("Risk Scoring", f"{risk_score}/100 — {risk_level}", "green" if risk_level == "GREEN" else ("yellow" if risk_level == "YELLOW" else "red")),
            ("Disease Lookup", f"{len(dataset_matches)} diseases matched", "green" if dataset_matches else "yellow"),
            ("RAG Retrieval", f"{len(retrieved)} chunks retrieved", "green" if retrieved else "yellow"),
            ("SOAP Generation", "Complete", "green"),
        ]

        for step_name, step_detail, step_color in pipeline_steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="dot dot-{step_color}"></div>
              <div style="flex:1">
                <span style="font-weight:600;font-size:0.85rem;color:var(--text-primary)">{step_name}</span>
                <span style="color:var(--text-secondary);font-size:0.8rem;margin-left:0.5rem">— {step_detail}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:0.9rem;font-size:0.82rem;color:var(--text-secondary)">
              <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:0.3rem">Original Input</div>
              {original_input or "—"}
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="background:var(--bg-secondary);border-radius:var(--radius);padding:0.9rem;font-size:0.82rem;color:var(--text-secondary)">
              <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:0.3rem">English (post-translation)</div>
              {english_text or "—"}
            </div>
            """, unsafe_allow_html=True)

    with dev_tab2:
        st.markdown('<div class="section-title">Pipeline Timing Breakdown</div>', unsafe_allow_html=True)

        timing_labels = {
            "transcription_ms": "Transcription",
            "translation_ms": "Translation",
            "classification_ms": "Classification",
            "extraction_ms": "Extraction",
            "risk_scoring_ms": "Risk Scoring",
            "disease_lookup_ms": "Disease Lookup",
            "rag_retrieval_ms": "RAG Retrieval",
            "soap_generation_ms": "SOAP Report",
            "total_ms": "TOTAL"
        }

        if timings:
            cols = st.columns(4)
            i = 0
            for key, label in timing_labels.items():
                if key in timings:
                    val = timings[key]
                    with cols[i % 4]:
                        border_color = "var(--cyan)" if key == "total_ms" else "var(--border-subtle)"
                        st.markdown(f"""
                        <div class="timing-pill" style="border:1px solid {border_color};margin-bottom:0.5rem">
                          <span class="timing-value" style="color:{'var(--cyan)' if key=='total_ms' else 'var(--text-primary)'}">{val}ms</span>
                          <span class="timing-label">{label}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    i += 1

            total = timings.get("total_ms", 0)
            st.markdown("<br>", unsafe_allow_html=True)
            if total > 0:
                st.markdown('<div class="section-title">Time Distribution</div>', unsafe_allow_html=True)
                render_keys = [k for k in timing_labels if k != "total_ms" and k in timings]
                for key in render_keys:
                    label = timing_labels[key]
                    val = timings[key]
                    pct = min(int((val / total) * 100), 100)
                    st.markdown(f"""
                    <div style="margin:0.35rem 0">
                      <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-secondary);margin-bottom:0.2rem">
                        <span>{label}</span>
                        <span style="font-family:var(--font-mono)">{val}ms ({pct}%)</span>
                      </div>
                      <div style="background:var(--bg-secondary);border-radius:999px;height:5px">
                        <div style="width:{pct}%;height:100%;background:var(--cyan);border-radius:999px;opacity:0.7"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No timing data available.")

    with dev_tab3:
        st.markdown('<div class="section-title">Raw Analysis JSON</div>', unsafe_allow_html=True)
        st.code(json.dumps(analysis, indent=2, ensure_ascii=False), language="json")

        st.markdown('<div class="section-title">Full Pipeline Response</div>', unsafe_allow_html=True)
        display_result = {k: v for k, v in result.items() if k != "soap_report"}
        st.code(json.dumps(display_result, indent=2, ensure_ascii=False), language="json")

        st.download_button(
            "Download Full JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="tabeebai_debug.json",
            mime="application/json"
        )

    with dev_tab4:
        st.markdown('<div class="section-title">RAG Retrieval Debug</div>', unsafe_allow_html=True)
        if retrieved:
            for i, chunk in enumerate(retrieved):
                st.markdown(f"""
                <div class="rag-chunk">
                  <div class="rag-title">
                    Chunk {i+1}: {chunk.get("title", "Untitled")}
                    <span class="rag-score">score: {chunk.get("score", 0):.4f}</span>
                  </div>
                  <div style="font-size:0.83rem;color:var(--text-secondary);line-height:1.65;margin-top:0.4rem">
                    {chunk.get("text", "")}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No RAG chunks retrieved.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Disease Lookup Debug</div>', unsafe_allow_html=True)
        if dataset_matches:
            st.code(json.dumps(dataset_matches, indent=2), language="json")
        else:
            st.info("No disease matches.")

        detail = analysis.get("dataset_match_detail", [])
        if detail:
            st.markdown('<div class="section-title">Match Detail Strings</div>', unsafe_allow_html=True)
            for item in detail:
                st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.8rem;color:var(--text-muted);padding:0.15rem 0">{item}</div>', unsafe_allow_html=True)

    with dev_tab5:
        st.markdown('<div class="section-title">Models Used</div>', unsafe_allow_html=True)
        models = result.get("models_used", {})
        if models:
            for role, model_name in models.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;background:var(--bg-secondary);border:1px solid var(--border-subtle);border-radius:var(--radius);padding:0.7rem 1rem;margin:0.3rem 0">
                  <span style="font-size:0.85rem;color:var(--text-secondary);text-transform:capitalize">{role.replace("_"," ")}</span>
                  <span style="font-family:var(--font-mono);font-size:0.78rem;color:var(--cyan);background:var(--cyan-dim);padding:0.2em 0.7em;border-radius:6px">{model_name}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Model info not available.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">System Health</div>', unsafe_allow_html=True)
        st.code(json.dumps(get_system_health(), indent=2), language="json")
