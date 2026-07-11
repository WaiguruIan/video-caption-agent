# Four Voices: Automated Video Captioning Agent

An intelligent, containerized AI agent that eliminates the operational bottlenecks of manual social media copy creation. Four Voices ingests raw video content and automatically generates four distinct narrative personas — each delivered in a structured, schema-compliant JSON format ready for production integration.

## The Multi-Model Pipeline
Data flows sequentially through specialized AI architectures to guarantee high-quality, contextual, and stylistically precise outputs:
*   **01 · Ingestion:** Headless video frame extraction handled natively via OpenCV.
*   **02 · Observation:** Kimi K2.6 runs multi-threaded, parallel frame analysis to concurrently extract visual semantics.
*   **03 · Grounding:** Gemini 2.5 Flash utilizes real-time Google Search tool execution to verify background data and eliminate LLM hallucinations.
*   **04 · Voice Generation:** DeepSeek-V4-Pro acts as the creative core, applying strict stylistic rubrics to generate 4 distinct narrative styles.

---

## The Four Voices
The agent evaluates the visual context and outputs the text across four highly targeted content perspectives:
1.  **Formal:** Clear, objective, and corporate-ready descriptions.
2.  **Sarcastic:** Witty, ironic, and high-engagement social copy.
3.  **Humorous Tech:** Platform-native engineering humor and contextual technical analogies.
4.  **Casual:** Relaxed, friendly, and highly relatable lifestyle copy.

---

## Project Structure
*   `main.py` — The core automated headless agent execution pipeline.
*   `app.py` — The interactive Streamlit user validation dashboard interface.
*   `Dockerfile` — Ultra-lightweight multi-stage isolated deployment configuration (`python:3.11-slim-bookworm`).
*   `.gitignore` / `.dockerignore` — Production-grade security constraints keeping environment keys, frames, and video assets isolated locally.

---

## Deployment & Execution (Track 2 Headless)

### 1. Interactive Web Interface
To run the evaluation dashboard locally:
```bash
pip install -r requirements.txt
streamlit run app.py
