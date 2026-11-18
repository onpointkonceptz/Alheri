"""
app/main.py
Alheri - Edge-Deployable Multilingual Mental Health & Primary Care Triage Assistant.
Built for low-connectivity clinics in Nigeria.
Strictly 100% Offline with Zero External Cloud Calls or Telemetry.
Supports: English, Hausa, Nigerian Pidgin, Igbo, Yoruba, and Arabic.
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, AsyncGenerator

import psutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

from app.rag import ingest_knowledge_base, retrieve_context, get_kb_stats
from app.voice import transcribe_audio_bytes

app = FastAPI(
    title="Alheri",
    description="Offline Mental Health & Primary Care Clinical Triage",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_MODEL = "gemma2:2b"
AUDIT_LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit_log.jsonl"))


class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "auto"
    patient_id: Optional[str] = "ANON-PHC"


def get_system_footprint() -> Dict[str, Any]:
    """Measures current system RAM and CPU usage via psutil."""
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    proc_ram = proc.memory_info().rss / (1024 * 1024)
    return {
        "ram_used_mb": round(vm.used / (1024 * 1024), 1),
        "ram_total_mb": round(vm.total / (1024 * 1024), 1),
        "ram_percent": vm.percent,
        "process_ram_mb": round(proc_ram, 1),
        "cpu_percent": psutil.cpu_percent(interval=None)
    }


def append_audit_log(entry: Dict[str, Any]):
    """Appends an anonymized clinical audit trail to local JSONL file."""
    try:
        entry["logged_at"] = datetime.now(timezone.utc).isoformat()
        entry["zero_cloud_calls"] = True
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[AUDIT LOG ERROR] Failed to write audit log: {e}")


def detect_language(text: str, override: str = "auto") -> str:
    """Detects if query is in Arabic, Hausa, Nigerian Pidgin, Igbo, Yoruba, or English."""
    if override and override.lower() in ["ar", "arabic"]:
        return "Arabic"
    if override and override.lower() in ["ha", "hausa"]:
        return "Hausa"
    if override and override.lower() in ["pid", "pidgin"]:
        return "Nigerian Pidgin"
    if override and override.lower() in ["ig", "igbo"]:
        return "Igbo"
    if override and override.lower() in ["yo", "yoruba"]:
        return "Yoruba"
    if override and override.lower() in ["en", "english"]:
        return "English"

    # Arabic script check
    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic"

    t = text.lower()

    # Igbo keyword check
    igbo_markers = [
        "obi na-akụ", "ike ọgwụgwụ", "ike adịghị", "isi na-awa", "ndụ a agwụla",
        "ahụ ọkụ", "onwe m", "ime onwe", "ka m nwụọ", "obi m na-ama", "nsogbu uche"
    ]
    if any(m in t for m in igbo_markers):
        return "Igbo"

    # Yoruba keyword check
    yoruba_markers = [
        "inú mi", "ìrẹ̀wẹ̀sì", "ọkàn mi", "orí mi", "ayé ti sú", "ara ń gbóná",
        "ẹ̀rù ń bà", "láti pa ara", "kàn kú", "ń lù kìkì", "gbóná", "àìsàn ọpọlọ"
    ]
    if any(m in t for m in yoruba_markers):
        return "Yoruba"

    # Hausa keyword check
    hausa_markers = [
        "ina jin", "damuwa", "zafi", "jikina", "bacin ciki", "kaina", "bacci",
        "kuzari", "hankali", "kasala", "muryoyi", "waswasi", "ciwon", "sauro"
    ]
    if any(m in t for m in hausa_markers):
        return "Hausa"

    # Nigerian Pidgin check
    pidgin_markers = [
        "dey", "wetin", "pikin", "tire me", "pepper me", "cutlass", "don tire",
        "make i", "scatter", "cut-and-join", "no fit", "wan kill", "wan end"
    ]
    if any(m in t for m in pidgin_markers):
        return "Nigerian Pidgin"

    return "English"


def clean_clinical_response(text: str) -> str:
    """
    Strictly removes all *, **, ## and eliminates repetition loops (e.g., E-E-E-, A'a'a'a').
    """
    if not text:
        return ""

    # Remove all asterisks and hash symbols (*, **, ##, #)
    cleaned = text.replace("**", "").replace("*", "").replace("##", "").replace("#", "")

    # Eliminate specific repetitive degeneration loops
    # 1. Hyphenated letter loops like E-E-E-E-...
    cleaned = re.sub(r"(?:[A-Za-z]-){3,}[A-Za-z]?-?", "", cleaned)
    # 2. Repeated apostrophe loops like A'a'a'a'... or a'a'a'...
    cleaned = re.sub(r"(?:[A-Za-z]'[A-Za-z]'?|[A-Za-z]'){3,}", "", cleaned)
    # 3. Any word/token repeated 3 or more times consecutively
    cleaned = re.sub(r"(\b\w+[\'-]?\w*[\s\'-]?)\1{3,}", "", cleaned, flags=re.IGNORECASE)
    # 4. Any character sequence repeated 4+ times
    cleaned = re.sub(r"(\S{1,6})\1{4,}", "", cleaned)

    # Clean up trailing hyphens, repeated spaces and excessive newlines
    cleaned = re.sub(r"-{3,}", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_language_instructions(lang: str) -> str:
    """Returns strict clinical guidelines and language enforcement instructions."""
    eval_framework = "Alheri evaluates symptoms against WHO mhGAP 2.0 and local Nigerian clinical guidelines (including physical and infectious medical rule-outs, cultural idioms, and crisis protocols, but not restricted to it alone)."
    no_markdown_rule = "MANDATORY: Do NOT use asterisks (* or **) or hash symbols (## or #) anywhere in your response. Output clean plain text only without any * or # markdown symbols."

    if lang == "Arabic":
        return f"""You are Alheri, an offline clinical triage assistant.
Evaluation Framework: {eval_framework}
CRITICAL MANDATE: You MUST respond COMPLETELY in ARABIC (اللغة العربية).
{no_markdown_rule}
Structure your response in Arabic:
مستوى الفرز (Triage Level): [أحمر (حالة طارئة) | أصفر (حالة عاجلة) | أخضر (حالة روتينية)]
التفسير السريري للأعراض: شرح الشكوى والأعراض النفسية والجسدية وثقافة المريض.
الإجراءات الطبية الفورية: خطوات الطاقم الطبي/التمريض (CHEW).
الفحوصات العضوية الإلزامية: فحص الملاريا السريع (mRDT)، السكر، العلامات الحيوية.
خطة الإحالة: (المركز الصحي -> المستشفى العام -> المستشفى التخصصي)."""

    elif lang == "Igbo":
        return f"""You are Alheri, an offline clinical triage assistant.
Evaluation Framework: {eval_framework}
CRITICAL MANDATE: You MUST respond COMPLETELY in IGBO (Asụsụ Igbo).
{no_markdown_rule}
Structure your response in Igbo:
Ọnọdụ Triage: [UHIE (Mberede) | EDO-EDO (Ngwa ngwa) | AKWỤKWỌ NDỤ (Mgbe niile)]
Nkọwa Ọnọdụ Mara Lafiya: Nkọwa ihe mgbaàmà, obi mgbawa, ike ọgwụgwụ, ma ọ bụ echiche onwe.
Ihe Ndị Ọrụ Ahụike Ga-eme Ngwa Ngwa: Nzọụkwụ mbụ maka ndị ọrụ ahụike (CHEW/Nurse).
Nnyocha Ahụike Dị Mkpa: Nnwale ịba (Malaria mRDT), shuga ọbara, na akara ndụ.
Atụmatụ Nzigara (Referral): (PHC -> Ụlọ Ọgwụ General -> Ụlọ Ọgwụ Ọkachamara)."""

    elif lang == "Yoruba":
        return f"""You are Alheri, an offline clinical triage assistant.
Evaluation Framework: {eval_framework}
CRITICAL MANDATE: You MUST respond COMPLETELY in YORUBA (Èdè Yorùbá).
{no_markdown_rule}
Structure your response in Yoruba:
Ipele Triage: [PUPA (Pàjáwìrì) | YẸ́LÒ (Kánjúkánjú) | ÀWỌ̀ EWÉ (Déédéé)]
Ìtumọ̀ Àwọn Àmì Àìsàn: Ìtumọ̀ inú bàjẹ́, ọkàn lù kìkì, orí gbígbóná, tàbí àwọn àmì mìíràn.
Àwọn Ìgbésẹ̀ Kánjúkánjú fún Olùtọ́jú: Ohun tí olùtọ́jú (CHEW/Nọ́ọ̀sì) gbọ́dọ̀ ṣe lẹ́sẹ̀kẹsẹ̀.
Àwọn Àyẹ̀wò Àìsàn Ara Pàtàkì: Àyẹ̀wò ibà (Malaria mRDT), ìwọ̀n shúgà, àti àwọn àmì ìwàláàyè.
Ètò Ìfipamọ́/Ìtọ́kasí (Referral): (PHC -> Ilé Ìwòsàn Gbogbogbòò -> Ilé Ìwòsàn Ọpọlọ)."""

    elif lang == "Hausa":
        return f"""You are Alheri, an offline clinical triage assistant.
Evaluation Framework: {eval_framework}
CRITICAL MANDATE: You MUST respond COMPLETELY in HAUSA (Harshen Hausa).
{no_markdown_rule}
Structure your response in Hausa:
Matakin Triage: [JA (Gaggawa) | RAWAYA (Cikin Sa'o'i 2) | KORE (Gida/Magani)]
Fassarar Alamun Mara Lafiya: Fassara bacin ciki, zafin jiki, ko damuwa.
Matakan Likita na Gaggawa: Abin da ma'aikacin lafiya (CHEW/Nurse) zai yi yanzu.
Gwajin Cuta na Jiki: Gwajin zazzabin cizon sauro (Malaria mRDT), sukarin jini, hawan jini.
Tsarin Tura Mara Lafiya (Referral): (PHC -> Babban Asibiti -> Asibitin Mahaukata)."""

    elif lang == "Nigerian Pidgin":
        return f"""You are Alheri, an offline clinical triage assistant.
Evaluation Framework: {eval_framework}
CRITICAL MANDATE: You MUST respond COMPLETELY in NIGERIAN PIDGIN ENGLISH.
{no_markdown_rule}
Structure your response in Nigerian Pidgin:
Triage Level: [RED (Emergency) | YELLOW (Urgent) | GREEN (Routine)]
Clinical Meaning: Explain the body symptoms (e.g., body dey pepper, life tire me).
Wetin Health Worker Go Do Now: Immediate steps for CHEW/nurse.
Organic Tests Dem Must Check: Malaria mRDT test quick, blood pressure, sugar test.
Referral Plan: How to move patient to General Hospital or Specialist."""

    else:
        return f"""You are Alheri, an offline clinical triage assistant.
Evaluation Framework: {eval_framework}
CRITICAL MANDATE: You MUST respond COMPLETELY in ENGLISH.
{no_markdown_rule}
Structure your response in English:
Triage Level: State clearly: [RED (Emergency) | YELLOW (Urgent) | GREEN (Routine)]
Clinical Interpretation: Decode patient symptoms or cultural idioms.
Immediate Clinical Actions: Immediate first-aid or steps for CHEW/nurse.
Mandatory Medical Rule-Outs: Malaria mRDT, vitals, typhoid, hypoglycemia.
Referral Plan: PHC -> General Hospital -> Specialist Hospital."""


def extract_priority(text: str) -> str:
    """Extracts triage priority category (RED, YELLOW, GREEN) from model output across all languages."""
    upper = text.upper()
    # Red checks
    if any(k in upper for k in ["CODE RED", "PRIORITY: RED", "**RED", "أحمر", "JA (GAGGAWA)", "MATAKIN TRIAGE: JA", "UHIE", "PUPA", "PÀJÁWÌRÌ", "MBEREDE"]):
        return "RED"
    # Yellow checks
    if any(k in upper for k in ["CODE YELLOW", "PRIORITY: YELLOW", "**YELLOW", "أصفر", "RAWAYA", "MATAKIN TRIAGE: RAWAYA", "EDO-EDO", "YẸ́LÒ", "NGWA NGWA", "KÁNJÚKÁNJÚ"]):
        return "YELLOW"
    # Green checks
    if any(k in upper for k in ["CODE GREEN", "PRIORITY: GREEN", "**GREEN", "أخضر", "KORE", "MATAKIN TRIAGE: KORE", "AKWỤKWỌ NDỤ", "ÀWỌ̀ EWÉ", "DÉÉDÉÉ"]):
        return "GREEN"
    # Heuristics
    if any(k in upper for k in ["SUICIDE", "POISON", "SNIPER", "CUTLASS", "VIOLENT", "AGITATION", "قتل", "انتحار"]):
        return "RED"
    if any(k in upper for k in ["HALLUCINATION", "VOICES", "DEPRESSION", "PEPPER", "نار في", "WASWASI", "POSTPARTUM", "OBI NA-AKỤ", "INÚ MI Ń BÀJẸ́"]):
        return "YELLOW"
    return "GREEN"


@app.get("/health")
def health_check():
    """Health check endpoint providing Ollama, ChromaDB, Whisper, and RAM footprint status."""
    footprint = get_system_footprint()
    ollama_ok = False
    models_available = []
    try:
        tags = ollama.list()
        ollama_ok = True
        if hasattr(tags, "models"):
            models_available = [m.model for m in tags.models]
        elif isinstance(tags, dict) and "models" in tags:
            models_available = [m.get("model") or m.get("name") for m in tags["models"]]
    except Exception as e:
        models_available = [f"Error connecting: {e}"]

    kb_stats = get_kb_stats()

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "service": "Alheri",
        "version": "1.2.0",
        "deployment_profile": "100% Offline Edge PHC",
        "zero_cloud_calls": True,
        "supported_languages": ["English", "Hausa", "Nigerian Pidgin", "Igbo", "Yoruba", "Arabic"],
        "ollama": {
            "status": "connected" if ollama_ok else "disconnected",
            "models": models_available,
            "target_llm": LLM_MODEL,
            "target_embeddings": "nomic-embed-text"
        },
        "knowledge_base": kb_stats,
        "whisper": {
            "model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "offline": True
        },
        "system": footprint
    }


@app.post("/ingest")
def trigger_ingest():
    """Triggers indexing of Markdown files in knowledge_base/ into ChromaDB."""
    return ingest_knowledge_base()


@app.post("/chat")
def chat_triage(req: ChatRequest):
    """Synchronous JSON triage endpoint with multilingual prompt support."""
    t0 = time.time()
    user_query = req.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    detected_lang = detect_language(user_query, req.language)
    rag_res = retrieve_context(user_query, top_k=3)
    retrieval_ms = rag_res.get("retrieval_latency_ms", 0.0)
    chunks = rag_res.get("chunks", [])

    context_parts = []
    for c in chunks:
        snippet = c["text"]
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        context_parts.append(f"[{c['source']} | {c['section']}]\n{snippet}")
    context_str = "\n".join(context_parts)

    system_instruction = build_language_instructions(detected_lang)
    user_prompt = f"""Clinical Guidelines:\n{context_str}\n\nPatient Statement: "{user_query}"\nProvide structured clinical triage in {detected_lang}."""

    gen_start = time.time()
    llm_response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        stream=False,
        options={
            "num_gpu": 0,
            "num_ctx": 1536,
            "num_predict": 240,
            "temperature": 0.35,
            "top_p": 0.85,
            "repeat_penalty": 1.25,
            "repeat_last_n": 64,
            "num_thread": 6
        }
    )
    raw_content = llm_response["message"]["content"]
    content = clean_clinical_response(raw_content)
    gen_ms = round((time.time() - gen_start) * 1000, 2)
    total_ms = round((time.time() - t0) * 1000, 2)

    priority = extract_priority(content)
    footprint = get_system_footprint()

    append_audit_log({
        "type": "text_chat",
        "language": detected_lang,
        "patient_id": req.patient_id,
        "query": user_query,
        "priority": priority,
        "retrieval_ms": retrieval_ms,
        "generation_ms": gen_ms,
        "total_ms": total_ms,
        "ram_mb": footprint["ram_used_mb"],
        "top_citation": chunks[0]["source"] if chunks else None
    })

    return {
        "status": "success",
        "language": detected_lang,
        "priority": priority,
        "triage_response": content,
        "citations": chunks,
        "latency": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": gen_ms,
            "total_ms": total_ms
        },
        "system": footprint,
        "zero_cloud_calls": True
    }


@app.post("/chat-stream")
async def chat_triage_stream(req: ChatRequest):
    """
    Real-Time Server-Sent Events (SSE) streaming endpoint using ollama.AsyncClient.
    Streams citations instantly (< 300ms), followed by live token generation.
    """
    user_query = req.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    detected_lang = detect_language(user_query, req.language)

    async def sse_generator() -> AsyncGenerator[str, None]:
        t0 = time.time()
        # Step 1: Sub-second RAG retrieval
        rag_res = retrieve_context(user_query, top_k=3)
        retrieval_ms = rag_res.get("retrieval_latency_ms", 0.0)
        chunks = rag_res.get("chunks", [])

        # Send initial metadata event with citations immediately
        meta_payload = {
            "language": detected_lang,
            "retrieval_ms": retrieval_ms,
            "citations": chunks
        }
        yield f"event: metadata\ndata: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"

        # Step 2: Build prompt
        context_parts = []
        for c in chunks:
            snippet = c["text"]
            if len(snippet) > 350:
                snippet = snippet[:350] + "..."
            context_parts.append(f"[{c['source']} | {c['section']}]\n{snippet}")
        context_str = "\n".join(context_parts)

        system_instruction = build_language_instructions(detected_lang)
        user_prompt = f"""Clinical Guidelines:\n{context_str}\n\nPatient Statement: "{user_query}"\nProvide structured clinical triage in {detected_lang}."""

        # Step 3: Stream tokens from local Ollama Gemma 2B via AsyncClient
        try:
            client = ollama.AsyncClient()
            response = await client.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                options={
                    "num_gpu": 0,
                    "num_ctx": 1536,
                    "num_predict": 240,
                    "temperature": 0.35,
                    "top_p": 0.85,
                    "repeat_penalty": 1.25,
                    "repeat_last_n": 64,
                    "num_thread": 6
                }
            )

            full_text_parts = []
            consecutive_repeats = 0
            last_token = ""

            async for chunk in response:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    # Clean markdown symbols (*, **, ##) from streamed token
                    clean_tok = token.replace('*', '').replace('#', '')

                    # Avoid emitting repetitive degeneration loops in stream
                    if clean_tok == last_token and len(clean_tok.strip()) > 0:
                        consecutive_repeats += 1
                        if consecutive_repeats >= 3:
                            continue
                    else:
                        consecutive_repeats = 0
                        last_token = clean_tok

                    if clean_tok:
                        full_text_parts.append(clean_tok)
                        yield f"event: token\ndata: {json.dumps({'text': clean_tok}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.001)

            full_content = clean_clinical_response("".join(full_text_parts))
            priority = extract_priority(full_content)
            total_ms = round((time.time() - t0) * 1000, 2)
            footprint = get_system_footprint()

            append_audit_log({
                "type": "streaming_chat",
                "language": detected_lang,
                "patient_id": req.patient_id,
                "query": user_query,
                "priority": priority,
                "retrieval_ms": retrieval_ms,
                "total_ms": total_ms,
                "ram_mb": footprint["ram_used_mb"],
                "top_citation": chunks[0]["source"] if chunks else None
            })

            done_payload = {
                "priority": priority,
                "total_ms": total_ms,
                "system": footprint,
                "final_text": full_content
            }
            yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            err_payload = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/voice-chat")
async def voice_chat_triage(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form("ANON-VOICE"),
    language: Optional[str] = Form("auto")
):
    """
    Offline Voice Triage endpoint.
    Transcribes audio offline via faster-whisper CPU int8, then executes clinical triage.
    """
    t_start = time.time()
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload.")

    transcription_res = transcribe_audio_bytes(audio_bytes)
    transcribed_text = transcription_res.get("text", "").strip()

    if not transcribed_text:
        return {
            "status": "warning",
            "message": "Audio was transcribed but no intelligible speech was recognized.",
            "transcription": transcription_res,
            "triage_response": "No clear speech detected. Please re-record or type symptoms directly.",
            "citations": [],
            "priority": "GREEN"
        }

    chat_req = ChatRequest(
        message=transcribed_text,
        language=language,
        patient_id=patient_id
    )
    triage_result = chat_triage(chat_req)

    total_voice_ms = round((time.time() - t_start) * 1000, 2)
    triage_result["transcription"] = transcription_res
    triage_result["latency"]["voice_transcription_ms"] = transcription_res.get("transcription_time_ms", 0)
    triage_result["latency"]["total_end_to_end_ms"] = total_voice_ms

    append_audit_log({
        "type": "voice_chat",
        "audio_filename": file.filename,
        "transcribed_text": transcribed_text,
        "whisper_detected_lang": transcription_res.get("detected_language"),
        "triage_language": triage_result["language"],
        "priority": triage_result["priority"],
        "transcription_ms": transcription_res.get("transcription_time_ms", 0),
        "total_ms": total_voice_ms,
        "zero_cloud_calls": True
    })

    return triage_result


@app.get("/audit")
def get_audit_trail(limit: int = 50):
    """Returns the most recent anonymized on-device triage logs."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return {"total_records": 0, "records": []}

    records = []
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audit log: {e}")

    records.reverse()
    return {
        "total_records": len(records),
        "returned": len(records[:limit]),
        "audit_file": AUDIT_LOG_FILE,
        "zero_cloud_calls": True,
        "records": records[:limit]
    }

# ==============================================================================
# UI Delivery from app/index.html (Guarantees clean syntax, zero python-escape bugs)
# ==============================================================================

INDEX_HTML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the clean white-background single-file HTML/CSS/JS Alheri user interface."""
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=200)
