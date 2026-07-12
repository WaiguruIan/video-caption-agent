import os
import re
import base64
import glob
import json
import time
import sys
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

# ==========================================
# 1. INITIALIZATION & CREDENTIALS
# ==========================================
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not FIREWORKS_API_KEY:
    sys.exit("❌ Critical Error: Missing FIREWORKS_API_KEY in environment variables.")

fireworks_client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKS_API_KEY,
)

# Gemini grounding is optional enrichment, not part of the harness-guaranteed stack.
GEMINI_ENABLED = False
gemini_client = None
GEMINI_MODEL = "gemini-2.5-flash"
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
    except Exception as e:
        print(f"⚠️ Gemini client failed to initialize ({e}). Continuing without factual grounding.")

MAX_VISION_WORKERS = 6
KIMI_RPM = 20
DEEPSEEK_RPM = 600


class RateLimiter:
    """Thread-safe sliding-window limiter. acquire() blocks until a slot is free."""
    def __init__(self, max_calls_per_minute):
        self.max_calls = max_calls_per_minute
        self.period = 60.0
        self.timestamps = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= self.period:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_calls:
                    self.timestamps.append(now)
                    return
                wait = self.period - (now - self.timestamps[0]) + 0.05
            time.sleep(max(wait, 0.05))


kimi_rate_limiter = RateLimiter(KIMI_RPM)
deepseek_rate_limiter = RateLimiter(DEEPSEEK_RPM)


def _is_retryable(e):
    """Shared retry check used for BOTH Fireworks and Gemini calls, for consistency."""
    status = getattr(e, "status_code", None)
    if status in [429, 500, 502, 503, 504]:
        return True
    msg = str(e).lower()
    return any(err in msg for err in ["429", "rate_limit", "connection", "timeout", "overloaded", "unavailable"])


def _backoff_sleep(base_delay):
    time.sleep(min(base_delay, 60.0) + random.uniform(0.1, 1.5))


# ==========================================
# UTILITY HELPERS
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def numerical_sort_key(path):
    numbers = re.findall(r"\d+", os.path.basename(path))
    return int(numbers[0]) if numbers else 0


def clean_json_string(raw_str):
    """Extracts valid JSON payload block using a non-greedy structural brace isolation."""
    if not raw_str:
        return ""
    match = re.search(r"(\{.*\})", raw_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def save_atomic_json(file_path, data):
    """Writes JSON payload atomically to prevent file corruption during disk or IO drops."""
    temp_path = f"{file_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, file_path)


CAPTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "formal": {"type": "string"},
        "sarcastic": {"type": "string"},
        "humorous_tech": {"type": "string"},
        "humorous_non_tech": {"type": "string"}
    },
    "required": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
    "additionalProperties": False
}

# Changing "strict" to False resolves inference engine compiler failures on Fireworks.
CAPTIONS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "captions_response",
        "strict": False,
        "schema": CAPTION_JSON_SCHEMA,
    },
}
VERIFIED_CAPTIONS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verified_captions_response",
        "strict": False,
        "schema": CAPTION_JSON_SCHEMA,
    },
}


def download_remote_video(url_or_path, task_idx=0):
    if isinstance(url_or_path, str) and url_or_path.startswith(("http://", "https://")):
        print(f"📥 External network path detected. Launching download stream: {url_or_path}")
        local_target = f"active_evaluation_asset_{task_idx}.mp4"
        try:
            response = requests.get(url_or_path, stream=True, timeout=90)
            response.raise_for_status()
            with open(local_target, "wb") as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
            print(f"✅ External target pulled down safely: local reference -> '{local_target}'")
            return local_target
        except Exception as pull_err:
            print(f"⚠️ Primary downloader stalled: {pull_err}. Invoking internal fallback driver...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url_or_path, local_target)
                return local_target
            except Exception as severe_err:
                print(f"❌ Pull engine collapsed entirely: {severe_err}.")
                raise RuntimeError(f"Failed to download evaluation asset: {url_or_path}")
    return url_or_path


# ==========================================
# FRAME EXTRACTION MODULE
# ==========================================
def extract_video_frames(video_path, output_folder, task_idx=0, interval=30):
    processed_path = download_remote_video(video_path, task_idx)

    if not os.path.exists(processed_path):
        print(f"❌ Error: video file '{processed_path}' does not exist on the local volume subsystem.")
        return False

    os.makedirs(output_folder, exist_ok=True)
    old_files = glob.glob(os.path.join(output_folder, "*"))
    for f in old_files:
        try:
            os.remove(f)
        except Exception:
            pass

    video = cv2.VideoCapture(processed_path)
    if not video.isOpened():
        print(f"❌ Error: Could not open video file context: {processed_path}.")
        return False

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        actual_interval = max(1, min(int(fps), 120))
    else:
        actual_interval = interval

    print(f"🎬 Slicing video target: '{processed_path}' at 1 frame per {actual_interval} ticks...")
    frame_count = 0
    saved_count = 0

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_count % actual_interval == 0:
                frame_name = os.path.join(output_folder, f"frame_{saved_count}.jpg")
                write_success = cv2.imwrite(frame_name, frame)
                if write_success:
                    saved_count += 1
            frame_count += 1
    finally:
        video.release()
        if processed_path == f"active_evaluation_asset_{task_idx}.mp4" and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except Exception:
                pass

    print(f"✅ Slicing completed. Saved {saved_count} frames safely.")
    return saved_count > 0


# ==========================================
# STEP 1: VISION TIMELINE ANALYSIS
# ==========================================
def _describe_frame(idx, path):
    try:
        base64_img = encode_image(path)
    except Exception as e:
        return idx, f"[Error loading image frame asset: {str(e)}]"

    max_retries = 5
    delay = 3.0

    for attempt in range(max_retries):
        kimi_rate_limiter.acquire()
        try:
            response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/kimi-k2p6",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe exactly what is visible. Keep it strictly under 2 sentences.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                            },
                        ],
                    }
                ],
                max_tokens=100,
            )
            description = response.choices[0].message.content.strip()
            description = re.sub(
                r"^(Let me analyze the image|Let break down what|In this image, we see).*?:\s*",
                "",
                description,
                flags=re.IGNORECASE,
            )
            return idx, description
        except Exception as e:
            if _is_retryable(e) and attempt < max_retries - 1:
                _backoff_sleep(delay)
                delay *= 2
            else:
                return idx, f"[description unavailable — vision engine exception: {str(e)}]"
    return idx, "[description unavailable — max retries exhausted]"


def generate_visual_timeline(frames_dir):
    print("\n🔍 Step 1: Processing video frames with Kimi K2.6 (Parallelized)...")
    frame_extensions = ["*.jpg", "*.jpeg", "*.png"]
    frame_paths = []
    for ext in frame_extensions:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))

    frame_paths.sort(key=numerical_sort_key)
    if not frame_paths:
        raise ValueError(f"No frames discovered inside location: '{frames_dir}'")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_VISION_WORKERS) as executor:
        futures = {
            executor.submit(_describe_frame, idx, path): idx
            for idx, path in enumerate(frame_paths)
        }
        for future in as_completed(futures):
            idx, description = future.result()
            results[idx] = description
            print(f"  └─ Parsed frame processing context index: {idx + 1}/{len(frame_paths)}")

    timeline_entries = [f"Frame {idx}: {results[idx]}" for idx in sorted(results.keys())]
    return "\n".join(timeline_entries)


# ==========================================
# STEP 2: GEMINI GROUNDING (Refined Fact Guard)
# ==========================================
def fetch_internet_context(visual_timeline):
    if not GEMINI_ENABLED:
        print("\n🌐 Step 2: Grounding skipped (GEMINI_API_KEY not configured).")
        return ""

    print("\n🌐 Step 2: Grounding timeline using Gemini Search Tooling...")
    
    prompt = f"""
    You are an objective fact-checking agent. Your task is to analyze the provided visual timeline extracted from a video clip and ground its observations in verified real-world facts using your search tool.

    Visual Timeline Data:
    {visual_timeline}

    Instructions:
    1. Scan the timeline systematically. Identify verified real-world elements: specific individuals, notable public events, exact consumer products, or software interfaces.
    2. For every identified entity, provide a concise, cross-verified factual summary explaining what it actually is based on search results.
    3. If an entity, person, or software interface cannot be explicitly verified via search, you must output exactly: "Cannot confirm entity identity." for that specific item. Do not speculate or guess.

    Output Format:
    - Core Subject/Event: [Verified name or description]
    - Key Entities Involved: [Verified names, products, or brands]
    - Factual Event Context: [Brief, objective background summary]
    """

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    max_retries = 3
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            if response.candidates:
                try:
                    text_content = response.text
                    if text_content:
                        return text_content.strip()
                except ValueError:
                    print("⚠️ Gemini response text retrieval failed due to content classification blockers.")
                    return ""
        except Exception as e:
            if _is_retryable(e) and attempt < max_retries - 1:
                _backoff_sleep(delay)
                delay *= 2
            else:
                print(f"⚠️ Gemini grounding failed ({e}). Continuing without it.")
                break
    return ""


# ==========================================
# STEP 3: DEEPSEEK COPYWRITING ENGINE WITH SCHEMA (Optimized Constraints)
# ==========================================
def generate_final_captions(visual_timeline, factual_context):
    print("\n🧠 Step 3: Generating structured payloads using DeepSeek-V4-Pro...")

    system_prompt = (
        "You are an advanced creative copywriting agent. Generate exactly 4 distinct caption styles "
        "as a valid JSON object matching the requested properties schema.\n\n"
        "CRITICAL MECHANICAL CONSTRAINTS:\n"
        "- Length: Every single caption style must be exactly one sentence. No multi-sentence blocks.\n"
        "- Punctuation & Style Rules: Do not use exclamation marks (!) or emojis anywhere in the JSON object.\n\n"
        "STRICT STYLE RUBRIC:\n"
        "- formal: Exactly 1 sentence. Purely descriptive tone. Do not use contractions (e.g., use 'is not' instead of 'isn't'). Do not use idioms or informal language.\n"
        "- sarcastic: Exactly 1 sentence. Must feature a deadpan contrast or ironic understatement. Period ending only.\n"
        "- humorous_tech: Exactly 1 sentence. Must bridge the visual action to a concrete systems-engineering analogy (e.g., database deadlocks, race conditions, memory leaks, null pointers).\n"
        "- humorous_non_tech: Exactly 1 sentence. Casual and conversational register. Contractions are allowed. Must use an everyday situational comparison.\n\n"
        "ACCURACY & GROUNDING RULES:\n"
        "- Rely strictly on the provided Timeline and Factual Context.\n"
        "- Do not invent background information, names, or metrics not explicitly present in the data.\n"
        "- Every caption variant must explicitly tie back to at least one concrete visual detail observed in the timeline."
    )

    factual_injection = f"Factual Context:\n{factual_context}\n\n" if factual_context else "Factual Context: [Not available. Rely entirely on the visual timeline.]\n\n"
    user_content = f"{factual_injection}Timeline:\n{visual_timeline}"

    max_retries = 4
    draft_delay = 4.0
    raw_captions = None

    for attempt in range(max_retries):
        deepseek_rate_limiter.acquire()
        try:
            # First try with JSON schema mode (strict set to False)
            response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=CAPTIONS_RESPONSE_FORMAT,
                max_tokens=1500,
                temperature=0.3,
            )
            raw_captions = response.choices[0].message.content.strip()
            if clean_json_string(raw_captions):
                break
        except Exception:
            # Fallback directly to regular json_object if schema parsing fails on the endpoint
            try:
                response = fireworks_client.chat.completions.create(
                    model="accounts/fireworks/models/deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": system_prompt + "\nOutput raw JSON only."},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=1500,
                    temperature=0.3,
                )
                raw_captions = response.choices[0].message.content.strip()
                if clean_json_string(raw_captions):
                    break
            except Exception as e:
                if _is_retryable(e) and attempt < max_retries - 1:
                    _backoff_sleep(draft_delay)
                    draft_delay *= 2
                else:
                    print(f"❌ DeepSeek Primary generation execution fault: {e}")
                    return None

    if not raw_captions or not clean_json_string(raw_captions):
        print("❌ Primary caption generation failed to yield a valid schema payload string.")
        return None

    print("🛡️ Running generated captions through the Verification Guard...")
    gate_system_prompt = (
        "You are an automated data verification gatekeeper. Analyze the target JSON captions against "
        "the raw Visual Timeline and Factual Context. Strip or fix any unconfirmed assertions or hallucinations. "
        "Ensure compliance with the mechanical constraints. Output JSON matching the validation schema rules."
    )

    gate_user_content = f"Timeline:\n{visual_timeline}\n\nContext:\n{factual_context}\n\nTarget JSON:\n{raw_captions}"

    gate_delay = 4.0
    for attempt in range(max_retries):
        deepseek_rate_limiter.acquire()
        try:
            gate_response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": gate_system_prompt},
                    {"role": "user", "content": gate_user_content},
                ],
                response_format=VERIFIED_CAPTIONS_RESPONSE_FORMAT,
                max_tokens=1500,
                temperature=0.1,
            )
            res_content = gate_response.choices[0].message.content.strip()
            if clean_json_string(res_content):
                return res_content
        except Exception:
            try:
                gate_response = fireworks_client.chat.completions.create(
                    model="accounts/fireworks/models/deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": gate_system_prompt + "\nOutput raw JSON only."},
                        {"role": "user", "content": gate_user_content},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=1500,
                    temperature=0.1,
                )
                res_content = gate_response.choices[0].message.content.strip()
                if clean_json_string(res_content):
                    return res_content
            except Exception as e:
                if _is_retryable(e) and attempt < max_retries - 1:
                    _backoff_sleep(gate_delay)
                    gate_delay *= 2
                else:
                    print(f"⚠️ Verification Guard failure: {e}. Falling back safely to raw primary captions.")
                    return raw_captions

    print("⚠️ Verification Guard failed to safely clear validation schema. Defaulting to raw primary captions.")
    return raw_captions


def process_single_video(video_path, frames_dir, task_idx=0):
    """Encapsulates execution per asset item with automatic storage containment guarantees."""
    if not extract_video_frames(video_path, frames_dir, task_idx=task_idx):
        print(f"❌ Skipping target: Frame extraction failed for {video_path}")
        return None

    try:
        timeline = generate_visual_timeline(frames_dir)
        if not timeline.strip():
            print(f"❌ Skipping target: Empty timeline generated for {video_path}")
            return None

        factual_context = fetch_internet_context(timeline)
        final_json_string = generate_final_captions(timeline, factual_context)

        if final_json_string:
            cleaned_json = clean_json_string(final_json_string)
            if cleaned_json:
                try:
                    return json.loads(cleaned_json)
                except json.JSONDecodeError as parse_err:
                    print(f"❌ Structural breakdown parsing isolated JSON structure: {parse_err}")
            else:
                print("❌ Post-processed target text content lacks structured braces.")
    except Exception as single_task_err:
        print(f"⚠️ Task processing execution anomaly encountered: {single_task_err}")
    finally:
        # Strict filesystem cleanup guard to protect volume storage thresholds across long batches
        if os.path.exists(frames_dir):
            for file in glob.glob(os.path.join(frames_dir, "*")):
                try:
                    os.remove(file)
                except Exception:
                    pass
            try:
                os.rmdir(frames_dir)
            except Exception:
                pass
    return None


# ==========================================
# PIPELINE EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    INPUT_JSON_PATH = os.environ.get("INPUT_JSON_PATH", "/input/tasks.json")
    OUTPUT_JSON_PATH = os.environ.get("OUTPUT_JSON_PATH", "/output/results.json")
    DEFAULT_VIDEO_FILE = os.environ.get("VIDEO_PATH", "sample.mp4")
    FRAMES_DIRECTORY = os.environ.get("FRAMES_DIRECTORY", "extracted_frames")

    print("🚀 Initializing Production Pipeline...")

    tasks = []
    is_batch = False

    if os.path.exists(INPUT_JSON_PATH):
        try:
            with open(INPUT_JSON_PATH, "r") as config_file:
                task_data = json.load(config_file)

            if isinstance(task_data, list):
                tasks = task_data
                is_batch = True
            elif isinstance(task_data, dict):
                tasks = [task_data]
        except Exception as json_err:
            print(f"⚠️ Task parsing alert: {json_err}. Defaulting to parameters.")

    if not tasks:
        tasks = [{"video_path": DEFAULT_VIDEO_FILE}]

    output_dir = os.path.dirname(OUTPUT_JSON_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_results = []

    try:
        for idx, task in enumerate(tasks):
            target_video = task.get("video_path") or task.get("video") or task.get("video_url") or DEFAULT_VIDEO_FILE
            print(f"\n🎯 Processing Task {idx + 1}/{len(tasks)}: '{target_video}'")

            task_frames_dir = f"{FRAMES_DIRECTORY}_{idx}" if is_batch else FRAMES_DIRECTORY
            result_payload = process_single_video(target_video, task_frames_dir, task_idx=idx)

            if result_payload:
                merged_item = {}
                if isinstance(task, dict):
                    for key, val in task.items():
                        if key not in ["video_path", "video", "video_url"]:
                            merged_item[key] = val

                merged_item.update(result_payload)
                final_results.append(merged_item)

                current_payload = final_results if is_batch else final_results[0]
                
                # Persist state atomically to completely guarantee checkpoint file integrity
                try:
                    save_atomic_json(OUTPUT_JSON_PATH, current_payload)
                    print(f"💾 Progressive checkpoint saved safely to '{OUTPUT_JSON_PATH}'")
                except Exception as io_err:
                    print(f"⚠️ Critical progressive state save stalled on IO: {io_err}")

        if not final_results:
            sys.exit("❌ Pipeline Terminated: Zero valid payloads generated.")

    except Exception as pipeline_error:
        sys.exit(f"❌ Pipeline Operational Collapse: {pipeline_error}")
