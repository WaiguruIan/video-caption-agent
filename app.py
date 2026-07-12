"""
Front end for the AMD Hackathon ACT II — Track 2 video captioning agent.
Run with:  streamlit run app.py
(Run it from the same folder as your .env — see setup note in the sidebar.)
"""

import os
import json
import glob
import tempfile

import streamlit as st

# ------------------------------------------------------------------
# Import the backend pipeline. Wrapped in try/except because the
# pipeline module exits at import time if FIREWORKS_API_KEY is missing —
# we want a friendly setup screen here instead of a raw traceback.
# ------------------------------------------------------------------
PIPELINE_READY = True
PIPELINE_ERROR = None
try:
    import main as rp
except SystemExit as e:
    PIPELINE_READY = False
    PIPELINE_ERROR = str(e)
except Exception as e:
    PIPELINE_READY = False
    PIPELINE_ERROR = f"Unexpected import error: {e}"

st.set_page_config(
    page_title="Four Voices — Video Captioning Agent",
    page_icon="🎬",
    layout="wide",
)

# ==========================================================
# GLOBAL STYLE
# ==========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0B0D10;
        color: #EDEEF0;
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

    /* hide default streamlit chrome for a cleaner presentation surface */
    #MainMenu, footer, header { visibility: hidden; }

    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        color: #8A8F98;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }
    .hero-sub {
        color: #B7BBC2;
        font-size: 1.02rem;
        max-width: 640px;
        margin-bottom: 1.4rem;
    }
    .voice-legend {
        display: flex;
        gap: 1.1rem;
        margin-bottom: 2.2rem;
        flex-wrap: wrap;
    }
    .voice-chip {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        color: #B7BBC2;
        letter-spacing: 0.03em;
    }
    .voice-dot {
        width: 9px; height: 9px; border-radius: 50%;
        display: inline-block;
    }

    .step-row { display: flex; gap: 1.4rem; margin: 1.6rem 0 2.4rem 0; flex-wrap: wrap; }
    .step-card {
        flex: 1; min-width: 200px;
        background: #14171C;
        border: 1px solid #262B33;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
    }
    .step-num {
        font-family: 'IBM Plex Mono', monospace;
        color: #4A6FA5;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }
    .step-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem; margin-bottom: 0.3rem; }
    .step-desc { color: #8A8F98; font-size: 0.86rem; line-height: 1.4; }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: #8A8F98;
        margin: 2.2rem 0 0.7rem 0;
    }

    .stButton>button {
        background: #EDEEF0;
        color: #0B0D10;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
    }
    .stButton>button:hover { background: #FFFFFF; color: #0B0D10; }

    .stTextInput>div>div>input {
        background: #14171C;
        color: #EDEEF0;
        border: 1px solid #262B33;
        font-family: 'IBM Plex Mono', monospace;
    }

    .frame-caption {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8A8F98;
        margin-top: 0.3rem;
        line-height: 1.35;
    }
    .setup-panel {
        background: #14171C;
        border: 1px solid #4A6FA5;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #B7BBC2;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ACCENTS = {
    "formal": "#4A6FA5",
    "sarcastic": "#C9A227",
    "humorous_tech": "#2DD4BF",
    "humorous_non_tech": "#FF6F91",
}
LABELS = {
    "formal": "FORMAL",
    "sarcastic": "SARCASTIC",
    "humorous_tech": "TECH",
    "humorous_non_tech": "HUMOROUS NON TECH",
}
FONT_STACKS = {
    "formal": "'IBM Plex Sans', sans-serif",
    "sarcastic": "'IBM Plex Sans', sans-serif",
    "humorous_tech": "'IBM Plex Mono', monospace",
    "humorous_non_tech": "'IBM Plex Sans', sans-serif",
}

DEMO_TIMELINE = (
    "Frame 0: A player in a blue kit sprints past a defender near the edge of the box.\n"
    "Frame 1: On-screen scoreboard reads 1-0, clock shows 87:15.\n"
    "Frame 2: Close-up of the player pointing to his ear toward the crowd after scoring."
)
DEMO_CONTEXT = (
    "- Core Subject/Event: Late-game go-ahead goal celebration in a football match.\n"
    "- Key Entities Involved: Cannot confirm entity identity.\n"
    "- Factual Event Context: A clip depicting a stoppage-time goal celebration, consistent with a "
    "viral \"ice in his veins\" style highlight format."
)
DEMO_CAPTIONS = {
    "formal": "The player scored in the 87th minute and celebrated by gesturing toward the crowd.",
    "sarcastic": "Sure, waltz past a defender and score with two minutes left, no big deal at all.",
    "humorous_tech": "That run had zero merge conflicts — straight to production past the defense.",
    "humorous_non_tech": "Bro really said \"watch this\" and just walked it in with the clock winding down.",
}

# ==========================================================
# HERO
# ==========================================================
st.markdown('<div class="hero-eyebrow">AMD DEVELOPER HACKATHON · ACT II · TRACK 2</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">One clip. Four voices.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Kimi K2.6 reads the video frame by frame, then DeepSeek-V4-Pro rewrites '
    'what it saw in four mechanically distinct registers — grounded in what actually happened on screen, '
    'not just restyled adjectives.</div>',
    unsafe_allow_html=True,
)

legend_html = '<div class="voice-legend">' + "".join(
    f'<div class="voice-chip"><span class="voice-dot" style="background:{ACCENTS[k]}"></span>{LABELS[k]}</div>'
    for k in ACCENTS
) + "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

st.markdown(
    """
    <div class="step-row">
      <div class="step-card">
        <div class="step-num">01</div>
        <div class="step-title">Ingest</div>
        <div class="step-desc">The clip is downloaded (or read locally) and sliced into one frame per second.</div>
      </div>
      <div class="step-card">
        <div class="step-num">02</div>
        <div class="step-title">Observe</div>
        <div class="step-desc">Kimi K2.6 describes every frame in parallel, reassembled into a strict chronological timeline.</div>
      </div>
      <div class="step-card">
        <div class="step-num">03</div>
        <div class="step-title">Voice</div>
        <div class="step-desc">DeepSeek-V4-Pro drafts all four styles, then a verification pass strips anything unconfirmed.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# SETUP GUARD
# ==========================================================
if not PIPELINE_READY:
    st.markdown(
        f"""
        <div class="setup-panel">
        ⚠️ Backend not ready: {PIPELINE_ERROR}<br><br>
        → Run <code>streamlit run app.py</code> from the same folder as your <code>.env</code> file.<br>
        → Confirm it contains a line exactly like <code>FIREWORKS_API_KEY=fw_xxxxxxxx</code> (no quotes).<br>
        → Gemini grounding is optional — the app still works with only <code>FIREWORKS_API_KEY</code> set.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ==========================================================
# INPUT
# ==========================================================
st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)

demo_mode = st.toggle(
    "🎭 Demo mode — show a pre-computed example instead of calling the live APIs",
    value=False,
    help="Use this as a safety net during the live pitch if the network or API quota misbehaves.",
)

col_a, col_b = st.columns([2, 1])
with col_a:
    video_input = st.text_input("Video path or URL", value="sample.mp4", disabled=demo_mode)
with col_b:
    uploaded = st.file_uploader("...or upload a clip", type=["mp4", "mov", "m4v"], disabled=demo_mode)

generate = st.button("Generate captions", type="primary")

# ==========================================================
# PIPELINE RUN
# ==========================================================
if generate:
    if demo_mode:
        st.markdown('<div class="section-label">Reasoning (simulated for demo)</div>', unsafe_allow_html=True)
        with st.expander("Visual timeline + factual context", expanded=False):
            st.code(DEMO_TIMELINE, language=None)
            st.text(DEMO_CONTEXT)
        result_payload = DEMO_CAPTIONS

    else:
        target_path = video_input
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False)
            tmp.write(uploaded.read())
            tmp.close()
            target_path = tmp.name

        frames_dir = "frontend_frames"
        result_payload = None

        with st.status("Running the pipeline...", expanded=True) as status:
            st.write("**01 · Ingest** — downloading/reading and slicing frames...")
            ok = rp.extract_video_frames(target_path, frames_dir, task_idx=0)
            if not ok:
                status.update(label="Frame extraction failed", state="error")
                st.error("Couldn't read that video. Check the path/URL and try again.")
                st.stop()

            st.write("**02 · Observe** — Kimi K2.6 is describing each frame...")
            timeline = rp.generate_visual_timeline(frames_dir)
            if not timeline.strip():
                status.update(label="Vision analysis returned nothing", state="error")
                st.error("No frame descriptions came back. Try a different clip.")
                st.stop()

            frame_paths = sorted(
                glob.glob(os.path.join(frames_dir, "*.jpg")),
                key=rp.numerical_sort_key,
            )
            if frame_paths:
                st.write("Sample frames:")
                cols = st.columns(min(6, len(frame_paths)))
                for i, col in enumerate(cols):
                    with col:
                        st.image(frame_paths[i], use_container_width=True)

            context = ""
            if rp.GEMINI_ENABLED:
                st.write("**Grounding** — checking for real-world context via search...")
                context = rp.fetch_internet_context(timeline)
            else:
                st.write("**Grounding** — skipped (optional step, no Gemini key set).")

            st.write("**03 · Voice** — drafting four caption styles, then verifying...")
            raw = rp.generate_final_captions(timeline, context)

            if not raw:
                status.update(label="Caption generation failed", state="error")
                st.error("The copywriting step didn't return a result. Check your Fireworks quota and retry.")
                st.stop()

            try:
                result_payload = json.loads(rp.clean_json_string(raw))
            except json.JSONDecodeError:
                status.update(label="Malformed JSON from the model", state="error")
                st.error("The model's response wasn't valid JSON. Try again — this is rare but not impossible.")
                st.stop()

            with st.expander("Visual timeline + factual context", expanded=False):
                st.code(timeline, language=None)
                if context:
                    st.text(context)

            status.update(label="Done", state="complete")

    # ==========================================================
    # RESULTS — the Voice Grid
    # ==========================================================
    if result_payload:
        st.markdown('<div class="section-label">Result — the voice grid</div>', unsafe_allow_html=True)

        cards_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
        for key in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]:
            text = result_payload.get(key, "—")
            color = ACCENTS[key]
            font = FONT_STACKS[key]
            safe_text_js = json.dumps(text)
            cards_html += f"""
            <div style="background:#14171C;border:1px solid #262B33;border-left:3px solid {color};
                        border-radius:10px;padding:18px 20px;position:relative;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:0.12em;
                          color:{color};margin-bottom:10px;">{LABELS[key]}</div>
              <div style="font-family:{font};font-size:1.02rem;line-height:1.5;color:#EDEEF0;
                          min-height:70px;">{text}</div>
              <button onclick='navigator.clipboard.writeText({safe_text_js})'
                      style="margin-top:12px;background:none;border:1px solid #262B33;color:#8A8F98;
                             border-radius:6px;padding:4px 10px;font-family:'IBM Plex Mono',monospace;
                             font-size:0.68rem;cursor:pointer;">
                copy
              </button>
            </div>
            """
        cards_html += "</div>"

        st.components.v1.html(
            f"<div style='font-family:sans-serif;'>{cards_html}</div>",
            height=430,
            scrolling=True,
        )

        st.download_button(
            "Download as JSON",
            data=json.dumps(result_payload, indent=2),
            file_name="captions.json",
            mime="application/json",
        )
