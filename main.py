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

GEMINI_ENABLED = False
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_ENABLED = True

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
    status = getattr(e, "status_code", None)
    if status in [429, 500, 502, 503, 504]:
        return True
    msg = str(e).lower()
    return any(err in msg for err in ["429", "rate_limit", "connection", "timeout", "overloaded"])

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
    """Extracts valid JSON payload block using structural brace isolation."""
    if not raw_str:
        return ""
    match = re.search(r"\{.*\}", raw_str, re.DOTALL)
    if match:
        return match.group(0).strip()
    return ""

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
    actual_interval = int(fps) if fps > 0 else interval

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
# STEP 2: GEMINI GROUNDING
# ==========================================
def fetch_internet_context(visual_timeline):
    if not GEMINI_ENABLED:
        print("\n🌐 Step 2: Grounding skipped (GEMINI_API_KEY not configured).")
        return ""

    print("\n🌐 Step 2: Grounding timeline using Gemini Search Tooling...")
    prompt = f"""
    You are an objective fact-checking agent. Analyze the provided timeline extracted from a video clip. Ground observations in verified facts using search.

    Visual Timeline Data:
    {visual_timeline}

    Instructions:
    1. Identify any verified elements: individuals, specific events, products, or software interfaces.
    2. Provide a brief, cross-verified factual summary of what that event actually is.
    3. If an element cannot be verified, explicitly state "Cannot confirm entity identity."

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
            if "429" in str(e) and attempt < max_retries - 1:
                _backoff_sleep(delay)
                delay *= 2
            else:
                break
    return ""

# ==========================================
# STEP 3: DEEPSEEK COPYWRITING ENGINE WITH SCHEMA
# ==========================================
def generate_final_captions(visual_timeline, factual_context):
    print("\n🧠 Step 3: Generating structured payloads using DeepSeek-V4-Pro...")

    system_prompt = (
        "You are an advanced creative copywriting agent. Generate exactly 4 caption styles "
        "as a valid JSON object matching the requested properties schema.\n\n"
        "STRICT STYLE RUBRIC:\n"
        "- formal: 1 sentence, no contractions, no exclamation marks, no emoji, no idioms. Purely descriptive.\n"
        "- sarcastic: 1 sentence, must contain at least one ironic understatement or deadpan contrast. No exclamation marks.\n"
        "- humorous_tech: 1 sentence, must contain exactly one concrete systems-engineering analogy mapping to the visual action.\n"
        "- humorous_non_tech: 1 sentence, casual register, contractions allowed, everyday situational comparison.\n\n"
        "ACCURACY RULES:\n"
        "- Only reference entities explicitly confirmed. Do not invent details.\n"
        "- Every caption must reference at least one concrete visual detail from the timeline."
    )

    factual_injection = f"Factual Context:\n{factual_context}\n\n" if factual_context else "Factual Context: [Not available. Rely entirely on the visual timeline.]\n\n"
    user_content = f"{factual_injection}Timeline:\n{visual_timeline}"
    
    max_retries = 4
    delay = 4.0
    raw_captions = None

    for attempt in range(max_retries):
        deepseek_rate_limiter.acquire()
        try:
            response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "captions_response",
                        "schema": CAPTION_JSON_SCHEMA
                    }
                },
                max_tokens=1500,
                temperature=0.3,
            )
            raw_captions = response.choices[0].message.content.strip()
            if clean_json_string(raw_captions):
                break
        except Exception as e:
            if _is_retryable(e) and attempt < max_retries - 1:
                _backoff_sleep(delay)
                delay *= 2
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

    for attempt in range(max_retries):
        deepseek_rate_limiter.acquire()
        try:
            gate_response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": gate_system_prompt},
                    {"role": "user", "content": gate_user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verified_captions_response",
                        "schema": CAPTION_JSON_SCHEMA
                    }
                },
                max_tokens=1500,
                temperature=0.1,
            )
            res_content = gate_response.choices[0].message.content.strip()
            if clean_json_string(res_content):
                return res_content
            else:
                print(f"⚠️ Verification Guard attempt {attempt + 1} yielded non-JSON text. Retrying...")
        except Exception as e:
            if _is_retryable(e) and attempt < max_retries - 1:
                _backoff_sleep(delay)
                delay *= 2
            else:
                print(f"⚠️ Verification Guard failure: {e}. Falling back safely to raw primary captions.")
                return raw_captions

    print("⚠️ Verification Guard failed to safely clear validation schema. Defaulting to raw primary captions.")
    return raw_captions

def process_single_video(video_path, frames_dir, task_idx=0):
    """Encapsulates execution per asset item to support batch safely."""
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
                with open(OUTPUT_JSON_PATH, "w") as f:
                    json.dump(current_payload, f, indent=4)
                print(f"💾 Progressive checkpoint saved safely to '{OUTPUT_JSON_PATH}'")

        if not final_results:
            sys.exit("❌ Pipeline Terminated: Zero valid payloads generated.")
            
    except Exception as pipeline_error:
        sys.exit(f"❌ Pipeline Operational Collapse: {pipeline_error}")
