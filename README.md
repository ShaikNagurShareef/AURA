# AURA (Intelligent Health Network)

A multimodal AI-driven healthcare platform designed for modern clinical workflows.

## The Team
- **Nagur Shareef Shaik**
- **Aasrith Mandava**
- **Sri Venkata Naga Sai Kakani**
- **Aasrith Karthikeya Gogineni**

---

## 🌟 Major Contribution: Smart Oculomics (IRIS)
Our flagship accomplishment is the **Smart Oculomics (IRIS)** engine. Instead of limiting retinal scans to purely ocular disease classification, we built state-of-the-art predictive models utilizing **Vision Transformers (ViT)** that empower IRIS to perform profound, systemic health diagnostics.

**Key Strength:** Diagnosing **not just ocular conditions**, but predicting **Systemic Health and Demographics** entirely from a retinal image obtained from a standard mobile device.

By analyzing the micro-vascular structures in the retina, IRIS predicts:
- **Demographics:** Age & Gender correlation.
- **Systemic Health Risks:** Diabetes, Hypertension, and overall Cardiovascular Risk.
- **Complications:** Acute Myocardial Infarction (AMI), Neuropathy, and Nephropathy.

This turns a smartphone into a non-invasive, highly accessible window into a patient’s holistic health profile.

## 🧭 COMPASS: Agentic RAG System
**COMPASS** is our dedicated Insurance Guide built as a robust **Agentic RAG (Retrieval-Augmented Generation)** system.
- It actively retrieves live US health insurance plans, ACA Marketplace data, Medicare/Medicaid eligibility, and geo-localized premiums by proxying our external [InsuCompass-API Node](https://huggingface.co/spaces/nagur-shareef-shaik/InsuCompass-API/tree/main).
- It seamlessly assists users who are completely new to health insurance in navigating deductibles, subsidies, and plans, falling back to an on-board Gemini Reasoning engine if the external API is unreachable.

## 🤖 The AURA Ecosystem of Assistants
Beyond IRIS and COMPASS, AURA features specialized agents acting as healthcare navigators to maintain personal health and encourage preventive care:
- **PRISM (Diagnostic):** Multimodal diagnostic assistant capable of synthesizing lab reports and clinical documentation.
- **SAGE (Wellbeing):** A mental health counsellor leveraging Cognitive Behavioural Therapy (CBT) protocols for emotional support.
- **APOLLO (Virtual Doctor):** Provides accessible symptom consultation, initial triage, and actionable immediate guidance.
- **NORA (Dietary Expert):** Focuses on preventive care through dynamic, personalized nutritional advice and diet plans.
- **VISTA (Visualisation):** Generates reports and data visualisations to help users track their systemic health data and biomarkers easily.
- **AURA Orchestrator:** The platform's master intelligence router. It parses multi-intent user queries and seamlessly delegates them to the correct specialist in the network.

---

## 📊 Technical Highlights & Benchmarks
To ensure clinical relevance and high accuracy, our predictive Vision Transformer (ViT) models have been assessed and bench-marked against leading healthcare datasets:
- **mBRSET:** A comprehensive dataset for general retinography and correlating systemic biomarkers.
- **MIMIC-CXR:** Broad-scale chest radiography reasoning for multimodal alignments.
- **DeepEyeNet & ROCO:** For extensive ocular abnormalities and general radiology image analysis.

### Performance Metrics & Real-World Impact
- **Impact:** We democratize advanced health screening. By enabling a mobile device to act as a diagnostic capture tool, patients in remote or under-resourced areas can receive systemic health screenings in seconds.
- **Outcomes:** 
  - Substantially reduced diagnosis latency.
  - Significantly lower cost barriers for pre-screening chronic widespread conditions like Diabetes and Hypertension.
  - Empowered decision making: The moment a systemic risk is detected, the AURA ecosystem automatically refers users to preventative care (NORA) or coverage planning (COMPASS), increasing active consumer participation in their own health maintenance.

---

## 🛠️ Scaffold, Setup & Installation

### Tech Stack
-   **Frontend:** React 18, TypeScript, Vite, Custom MNC-grade Vanilla CSS architecture.
-   **Backend:** Python 3.11, FastAPI, Pydantic, Google Gemini 1.5 Pro, ChromaDB (Local persistent Vector Memory per user), SQLite (Relational Data & Threads).

### 1. Pre-requisites
-   Python 3.10+
-   Node.js 18+ & npm

### 2. Environment Configuration
Navigate to the `backend` directory, duplicate `.env.example` to `.env`, and populate it:
```dotenv
# backend/.env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Quick Start (Automated Setup)
We provide an interactive `start.sh` script that handles the entire installation framework for UNIX environments. 

From the root directory, grant execution permissions and run:
```bash
chmod +x start.sh
./start.sh
```

**What the script does automatically:**
1. **Backend Validation:** Checks for the `backend/.venv`. If missing, creates the virtual environment and automatically runs `pip install -r requirements.txt`.
2. **Frontend Validation:** Checks for `frontend-react/node_modules`. If missing, runs `npm install`.
3. **Concurrent Launch:** Boots the FastAPI backend on `http://localhost:8000` and the Vite React UI on `http://localhost:5173`.

### 4. Manual Start (Alternative)
**Backend:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend-react
npm install
npm run dev
```

---

## Contact
Email: aura@gmail.com
