# Alheri (الخير) — Edge-Deployable Clinical Triage Assistant

> **Alheri** *(Hausa & Arabic: "Grace", "Goodness", "Blessing")* is an edge-deployable, 100% offline multilingual mental health and primary care triage assistant designed specifically for low-connectivity, resource-constrained Primary Health Care Centres (PHCs) in Nigeria.

---

## 1. Human-Scale AI Principles

Alheri is engineered around three foundational **Human-Scale AI** principles:

### 🔬 Small
- **No Cloud Bloat**: Operates entirely without high-bandwidth cloud APIs, gigabit connections, or multi-billion parameter foundation models.
- **Micro Hardware Budget**: Runs comfortably on consumer CPU hardware (Mini PC, Intel N100, Raspberry Pi 5 8GB, or refurbished clinic laptops) with a **< 20W power draw**, making it ideal for solar-powered clinics with intermittent electrical grid supply.
- **Lean Memory Footprint**: Uses `gemma2:2b` (2.6B parameter model at Q4_0 quantization, ~1.6 GB) and `nomic-embed-text` (137M parameters, ~274 MB) via Ollama, plus `faster-whisper` (base int8, ~140 MB).

### 🎯 Intentional
- **Grounding Over Hallucination**: Rather than trusting general-purpose model weights for clinical decisions, Alheri forces strict grounding against curated Markdown protocols:
  1. `knowledge_base/mhgap_triage_basics.md` — WHO mhGAP-IG 2.0 adapted for non-specialist Community Health Extension Workers (CHEWs).
  2. `knowledge_base/hausa_pidgin_glossary.md` — Somatosensory and psychiatric idioms in Hausa (*Harshen Hausa*) and Nigerian Pidgin.
  3. `knowledge_base/crisis_escalation_protocol.md` — Emergency triage thresholds (Code Red, Yellow, Green), organic rule-outs, de-escalation, and referral pathways.
- **Clinician-Editable**: Local nurses, CHEWs, and medical officers can open and edit the Markdown files in any text editor without retraining neural networks or running code deployments.

### 🇳🇬 Tailored
- **Cultural & Idiomatic Competence**: Translates local idioms of distress across English, Hausa (*Harshen Hausa*), Nigerian Pidgin, Igbo (*Asụsụ Igbo*), Yoruba (*Èdè Yorùbá*), and Arabic (*العربية*) into formal clinical diagnostic criteria.
- **Language Matching Guarantee**: Automatically detects the query language and guarantees that all clinical responses, triage categories, and organic rule-outs are formulated in the **exact same language** as the inquiry.
- **Evaluation Scope**: Alheri evaluates symptoms against WHO mhGAP 2.0 and local Nigerian clinical guidelines (including physical and infectious medical rule-outs, cultural idioms, and crisis protocols, but not restricted to it alone).
- **Mandatory Organic Rule-Outs**: Prioritizes Nigerian primary care reality where psychiatric symptoms often mask life-threatening infections (e.g., cerebral malaria, typhoid psychosis, severe anemia, postpartum eclampsia, hypoglycemia).
- **Localized Referral Pathways**: Explicit escalation chain: PHC → General/District Hospital → Federal Neuropsychiatric Hospital (FNPH).

---

## 2. Architecture & Offline Guarantee

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                       ALHERI LOCAL EDGE SYSTEM (100% OFFLINE)                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  [Clinician Web UI]  ──►  [FastAPI Backend: app/main.py (Port 8000)]           │
│     (Single-file HTML)        │                                                │
│                               ├──► [/voice-chat] ──► faster-whisper (CPU int8) │
│                               │                                                │
│                               ├──► [/chat] ────────► app/rag.py                │
│                               │                           │                    │
│                               │           ┌───────────────┴───────────────┐    │
│                               │           ▼                               ▼    │
│                               │     ChromaDB Store                Ollama (Local)│
│                               │   (./chroma_store)             (localhost:11434)│
│                               │   hnsw:space=cosine                   │        │
│                               │           ▲                           ├── nomic-embed-text
│                               │           │ (Ingest)                  └── gemma2:2b
│                               │   knowledge_base/*.md                          │
│                               │                                                │
│                               └──► Local Audit Trail (audit_log.jsonl)         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Zero Cloud Calls & Zero Telemetry
- **No External Network Calls**: Telemetry is disabled in ChromaDB, Ollama runs entirely on `localhost:11434`, and faster-whisper runs on local CPU.
- **Air-Gapped Operation**: Once initial model weights are downloaded, Alheri can be severed from the internet completely.
- **On-Device Audit Trail**: Every triage query and latency measurement is logged strictly to a local file (`audit_log.jsonl`) with zero data leaving the clinic.

---

## 3. Directory Structure

```
Alheri/
├── knowledge_base/
│   ├── mhgap_triage_basics.md          # Non-specialist WHO mhGAP 2.0 triage protocol
│   ├── hausa_pidgin_glossary.md        # Hausa & Nigerian Pidgin idioms of distress
│   └── crisis_escalation_protocol.md   # Code Red/Yellow/Green referral & escalation
├── app/
│   ├── __init__.py                     # Package initialization
│   ├── rag.py                          # ChromaDB vector store + nomic-embed-text
│   ├── voice.py                        # faster-whisper offline speech-to-text (int8)
│   └── main.py                         # FastAPI web service & single-page UI
├── chroma_store/                       # Persistent on-disk vector database
├── benchmark.py                        # 10-case offline latency and RAM benchmark
├── BENCHMARKS.md                       # Output benchmark execution results
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation & deployment guide
```

---

## 4. Hardware Footprint & Clinic Deployment

| Component | Disk Footprint | Peak RAM Footprint | Processor Requirements |
|---|---|---|---|
| **Ollama Server** | ~50 MB | ~50 MB | Any x86_64 or ARM64 CPU |
| **gemma2:2b (Q4_0)** | 1.63 GB | ~1.9 GB VRAM/RAM | 2+ CPU cores |
| **nomic-embed-text** | 274 MB | ~340 MB RAM | Shared with LLM |
| **faster-whisper (base int8)** | 140 MB | ~250 MB RAM | Single CPU core |
| **ChromaDB + FastAPI** | < 100 MB | ~120 MB RAM | Single thread |
| **Total System Stack** | **~2.2 GB Disk** | **~2.8 - 3.5 GB RAM** | **Low-cost CPU / 8 GB RAM** |

### Recommended Solar/Battery Edge Appliances
1. **Intel N100 Mini PC** (4 cores, 8GB or 16GB DDR5, 128GB SSD) — typical consumption **6W - 15W** (can run 12+ hours on a modest 100Ah solar clinic battery).
2. **Raspberry Pi 5 (8GB RAM)** — running Linux ARM64 with Ollama.
3. **Refurbished Laptops (Intel Core i5 8th gen+)** — provides built-in keyboard, screen, and battery backup.

---

## 5. Quickstart & Installation

### Step 1: Install Ollama & Pull Models
```bash
# 1. Start Ollama
ollama serve

# 2. Pull local models (one-time setup)
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

### Step 2: Set Up Python Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Ingest the Clinical Knowledge Base
```bash
python -c "from app.rag import ingest_knowledge_base; print(ingest_knowledge_base())"
```

### Step 4: Run the Benchmark Suite
```bash
python benchmark.py
```
This executes the 10 English, Hausa, and Pidgin clinical scenarios and generates `BENCHMARKS.md`.

### Step 5: Start the Alheri Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open your browser to: **`http://localhost:8000`**

---

## 6. API Reference

### `GET /health`
Returns health status, local model availability, ChromaDB chunk count, and psutil RAM metrics.

### `POST /ingest`
Scans `knowledge_base/*.md`, chunks documents into semantic sections, embeds them using `nomic-embed-text`, and saves them to `./chroma_store`.

### `POST /chat-stream` (Real-Time SSE Streaming)
Accepts `{"message": "string", "language": "auto|en|ha|pid|ar", "patient_id": "string"}`. Streams Server-Sent Events (`metadata`, `token`, `done`) in real-time. Immediately sends grounded citations (< 300ms) and streams tokens live in the matching query language, completely eliminating loading latency.

### `POST /chat`
Accepts `{"message": "string", "language": "auto|en|ha|pid|ar", "patient_id": "string"}`. Returns synchronous JSON triage assessment with matched citations and latency breakdown.

### `POST /voice-chat`
Multipart form upload (`file: UploadFile`). Transcribes audio using `faster-whisper` and immediately runs clinical triage on the transcribed text in the detected or specified language.

### `GET /audit`
Returns the recent on-device audit log from `audit_log.jsonl`.
