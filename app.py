import os
import sys
import time
import gc
import hashlib
import json
import re
import threading
import subprocess
import torch
import numpy as np
import whisper
import uuid
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename

DEFAULT_MODEL = "medium"
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")

# --- مدل فارسی (Hugging Face) ------------------------------------------------
PERSIAN_MODEL_ID = "persian"
PERSIAN_MODEL_REPO = "givia/whisper-large-v3-turbo-persian"
# revision را پین می‌کنیم تا لیست فایل‌ها و حجم‌ها ثابت بماند.
PERSIAN_MODEL_REVISION = "9626d38558f04c02f480b702c9db740eea567911"
# (نام فایل، حجم دقیق، sha256 برای فایل بزرگ)
PERSIAN_MODEL_FILES = (
    ("config.json", 1225, None),
    ("generation_config.json", 3749, None),
    ("preprocessor_config.json", 340, None),
    ("tokenizer_config.json", 2111, None),
    ("tokenizer.json", 3931381, None),
    ("vocab.json", 1036558, None),
    ("merges.txt", 493869, None),
    ("added_tokens.json", 34648, None),
    ("special_tokens_map.json", 2186, None),
    ("normalizer.json", 52666, None),
    (
        "model.safetensors",
        3235581408,
        "93b2748f6a8b3c8adf9315879896aaa00f70f6a6a8793adfc7ef53021c5cdf19",
    ),
)

# مدل‌های قابل انتخاب. برای مدل‌های OpenAI، size با Content-Length رسمی همان
# فایل است و sha256 هم بخشی از URL رسمی Whisper است.
_KNOWN_MODELS = (
    {
        "id": "medium",
        "label": "Medium",
        "description": "سریع و متعادل",
        "backend": "whisper",
        "size": 1528008539,
    },
    {
        "id": "large-v3",
        "label": "High",
        "description": "دقیق‌ترین خروجی عمومی",
        "backend": "whisper",
        "size": 3087371615,
    },
    {
        "id": PERSIAN_MODEL_ID,
        "label": "Persian",
        "description": "بهترین کیفیت فارسی",
        "backend": "transformers",
        "repo_id": PERSIAN_MODEL_REPO,
        "revision": PERSIAN_MODEL_REVISION,
        "files": PERSIAN_MODEL_FILES,
        "size": sum(f[1] for f in PERSIAN_MODEL_FILES),
    },
)

MODEL_SPECS = {}
for _model in _KNOWN_MODELS:
    _model_id = _model["id"]
    if _model["backend"] == "whisper":
        _url = whisper._MODELS.get(_model_id)
        if not _url:
            continue
        _spec = dict(_model)
        _spec.update({
            "url": _url,
            "sha256": _url.rstrip("/").split("/")[-2],
            "path": os.path.join(MODEL_CACHE_DIR, f"{_model_id}.pt"),
        })
    else:
        _spec = dict(_model)
        _spec["path"] = os.path.join(MODEL_CACHE_DIR, _model_id)
    MODEL_SPECS[_model_id] = _spec

if DEFAULT_MODEL not in MODEL_SPECS:
    raise RuntimeError("The Whisper medium model is not available in this installation.")

MODEL_DOWNLOAD_STATE = {
    model_id: {"state": "idle", "error": None}
    for model_id in MODEL_SPECS
}
MODEL_DOWNLOAD_STATE_LOCK = threading.Lock()
MODEL_DOWNLOAD_LOCKS = {
    model_id: threading.Lock()
    for model_id in MODEL_SPECS
}


def _hf_ready_path(spec):
    return os.path.join(spec["path"], ".ready")


def _prepare_hf_model_dir(model_dir):
    """سازگار کردن چک‌پوینت‌های ذخیره‌شده با transformers 5.x برای transformers 4.x.

    دو تفاوت رایج:
    - extra_special_tokens در tokenizer_config به صورت list ذخیره شده است
      (نسخه‌های 4.x فقط dict را قبول می‌کنند).
    - processor_config.json به شکل تودرتو ذخیره شده است و AutoProcessor
      را دچار تداخل feature_extractor می‌کند؛ با حذف آن،
      preprocessor_config.json به‌درستی استفاده می‌شود.
    """
    tokenizer_config = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config):
        try:
            with open(tokenizer_config, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("extra_special_tokens"), list):
                data.pop("extra_special_tokens", None)
                with open(tokenizer_config, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print("Patched tokenizer_config.json for transformers 4.x", flush=True)
        except Exception as e:
            print(f"Warning: could not patch tokenizer_config.json: {e}", flush=True)

    processor_config = os.path.join(model_dir, "processor_config.json")
    if os.path.exists(processor_config):
        try:
            os.replace(processor_config, processor_config + ".bak")
            print("Moved processor_config.json aside (transformers 5.x format)", flush=True)
        except OSError:
            pass


def model_is_installed(model_id):
    spec = MODEL_SPECS.get(model_id)
    if not spec:
        return False

    if spec.get("backend") == "transformers":
        path = spec["path"]
        if not os.path.isdir(path) or not os.path.exists(_hf_ready_path(spec)):
            return False
        try:
            for filename, expected_size, _ in spec["files"]:
                file_path = os.path.join(path, filename)
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    return False
                # tokenizer_config بعد از پچ شدن حجمش کمی تغییر می‌کند.
                if filename != "tokenizer_config.json" and os.path.getsize(file_path) != expected_size:
                    return False
            return True
        except OSError:
            return False

    path = spec["path"]
    try:
        return os.path.exists(path) and os.path.getsize(path) >= spec["size"]
    except OSError:
        return False


def model_downloaded_bytes(model_id):
    spec = MODEL_SPECS.get(model_id)
    if not spec:
        return 0

    if spec.get("backend") == "transformers":
        total = 0
        for filename, expected_size, _ in spec["files"]:
            final_path = os.path.join(spec["path"], filename)
            part_path = final_path + ".part"
            try:
                if os.path.exists(final_path):
                    total += min(os.path.getsize(final_path), expected_size)
                elif os.path.exists(part_path):
                    total += min(os.path.getsize(part_path), expected_size)
            except OSError:
                pass
        return min(total, spec["size"])

    sizes = []
    for candidate in (spec["path"], spec["path"] + ".part"):
        try:
            if os.path.exists(candidate):
                sizes.append(os.path.getsize(candidate))
        except OSError:
            pass
    return min(max(sizes, default=0), spec["size"])


def set_model_download_state(model_id, state, error=None):
    with MODEL_DOWNLOAD_STATE_LOCK:
        MODEL_DOWNLOAD_STATE[model_id]["state"] = state
        MODEL_DOWNLOAD_STATE[model_id]["error"] = error


def sha256_of_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one_file(url, dest, expected_size, sha256=None, label="", on_verify=None):
    """دانلود یک فایل با resume؛ فایل نهایی فقط پس از تأیید حجم/هش ساخته می‌شود."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"

    # فایل نهایی اگر از قبل کامل است، دوباره دانلود نشود (در صورت وجود هش، بررسی شود).
    if os.path.exists(dest):
        try:
            if os.path.getsize(dest) == expected_size:
                if not sha256:
                    return
                if on_verify:
                    on_verify()
                if sha256_of_file(dest) == sha256:
                    return
                os.remove(dest)
            else:
                os.remove(dest)
        except OSError:
            pass

    if os.path.exists(part):
        try:
            if os.path.getsize(part) > expected_size:
                os.remove(part)
        except OSError:
            pass

    last_size = -1
    stagnant = 0
    for attempt in range(1, 121):
        cmd = [
            "curl", "-f", "-s", "-L", "-C", "-",
            "--connect-timeout", "15",
            "--speed-time", "60", "--speed-limit", "1024",
            "--retry", "2", "--retry-delay", "2",
            "-o", part, url,
        ]
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            raise RuntimeError("curl was not found; it is required to download models.")

        size = os.path.getsize(part) if os.path.exists(part) else 0
        if size > expected_size:
            # احتمالاً سرور Range را نادیده گرفته و فایل را از ابتدا نوشته است.
            try:
                os.remove(part)
            except OSError:
                pass
            size = 0
        if size >= expected_size:
            break

        if size == last_size:
            stagnant += 1
        else:
            stagnant = 0
            last_size = size
        if stagnant >= 5:
            raise RuntimeError(
                f"Download stalled for {label or os.path.basename(dest)} "
                f"({size}/{expected_size} bytes)."
            )

        print(
            f"Download {label or os.path.basename(dest)}: {size}/{expected_size} bytes "
            f"(attempt {attempt}/120), retrying...",
            flush=True,
        )
        time.sleep(1)

    size = os.path.getsize(part) if os.path.exists(part) else 0
    if size < expected_size:
        raise RuntimeError(
            f"Download incomplete for {label or os.path.basename(dest)} "
            f"({size}/{expected_size} bytes)."
        )

    if sha256:
        if on_verify:
            on_verify()
        digest = sha256_of_file(part)
        if digest != sha256:
            try:
                os.remove(part)
            except OSError:
                pass
            raise RuntimeError(f"Checksum mismatch for {label or os.path.basename(dest)}.")

    os.replace(part, dest)


def _download_transformers_model(model_id, spec):
    """دانلود چندفایلی مدل Hugging Face + برچسب نصب."""
    with MODEL_DOWNLOAD_LOCKS[model_id]:
        if model_is_installed(model_id):
            set_model_download_state(model_id, "completed")
            return True

        os.makedirs(spec["path"], exist_ok=True)
        ready = _hf_ready_path(spec)
        if os.path.exists(ready):
            try:
                os.remove(ready)
            except OSError:
                pass

        set_model_download_state(model_id, "downloading")
        print(
            f"Downloading model '{model_id}' from Hugging Face "
            f"({len(spec['files'])} files)...",
            flush=True,
        )
        try:
            for index, (filename, expected_size, sha256) in enumerate(spec["files"], 1):
                dest = os.path.join(spec["path"], filename)
                url = (
                    f"https://huggingface.co/{spec['repo_id']}/resolve/"
                    f"{spec['revision']}/{filename}"
                )
                print(
                    f"Model '{model_id}' file {index}/{len(spec['files'])}: {filename}",
                    flush=True,
                )

                def _on_verify():
                    set_model_download_state(model_id, "verifying")

                _download_one_file(
                    url,
                    dest,
                    expected_size,
                    sha256,
                    label=f"{model_id}/{filename}",
                    on_verify=_on_verify,
                )
                if MODEL_DOWNLOAD_STATE[model_id]["state"] == "verifying":
                    set_model_download_state(model_id, "downloading")

            _prepare_hf_model_dir(spec["path"])
            with open(ready, "w", encoding="utf-8") as f:
                json.dump(
                    {"repo_id": spec["repo_id"], "revision": spec["revision"]},
                    f,
                    ensure_ascii=False,
                )
            set_model_download_state(model_id, "completed")
            print(f"Model '{model_id}' downloaded successfully.", flush=True)
            return True
        except Exception as e:
            set_model_download_state(model_id, "error", str(e))
            print(f"Model download failed ({model_id}): {e}", flush=True)
            raise


def download_model_sync(model_id):
    """دانلود (یا ادامه دانلود) یک مدل همراه با بررسی SHA256."""
    spec = MODEL_SPECS.get(model_id)
    if not spec:
        raise ValueError(f"Unknown model: {model_id}")

    if spec.get("backend") == "transformers":
        return _download_transformers_model(model_id, spec)

    with MODEL_DOWNLOAD_LOCKS[model_id]:
        if model_is_installed(model_id):
            set_model_download_state(model_id, "completed")
            return True

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        dest = spec["path"]
        part = dest + ".part"
        total = spec["size"]

        # فایل نهایی ناقص/خراب را کنار می‌گذاریم تا از .part ادامه دهیم.
        if os.path.exists(dest) and not model_is_installed(model_id):
            try:
                os.remove(dest)
            except OSError:
                pass
        if os.path.exists(part):
            try:
                if os.path.getsize(part) > total:
                    os.remove(part)
            except OSError:
                pass

        set_model_download_state(model_id, "downloading")
        print(f"Downloading Whisper '{model_id}' model...", flush=True)
        try:
            for attempt in range(1, 121):
                cmd = [
                    "curl", "-f", "-s", "-L", "-C", "-",
                    "--connect-timeout", "15",
                    "--speed-time", "60", "--speed-limit", "1024",
                    "--retry", "2", "--retry-delay", "2",
                    "-o", part, spec["url"],
                ]
                try:
                    subprocess.run(cmd, check=False)
                except FileNotFoundError:
                    raise RuntimeError("curl was not found; it is required to download models.")

                size = os.path.getsize(part) if os.path.exists(part) else 0
                if size > total:
                    # احتمالاً سرور Range را نادیده گرفته و فایل دوباره نوشته شده است.
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                    size = 0
                if size >= total:
                    break
                print(
                    f"Model '{model_id}' download: {size}/{total} bytes "
                    f"(attempt {attempt}/120), retrying...",
                    flush=True,
                )
                time.sleep(2)

            size = os.path.getsize(part) if os.path.exists(part) else 0
            if size < total:
                raise RuntimeError(f"Download incomplete after retries ({size}/{total} bytes).")

            set_model_download_state(model_id, "verifying")
            print(f"Verifying checksum for '{model_id}' (this may take a moment)...", flush=True)
            digest = sha256_of_file(part)
            if digest != spec["sha256"]:
                try:
                    os.remove(part)
                except OSError:
                    pass
                raise RuntimeError("Checksum mismatch; the downloaded model file is corrupt.")

            os.replace(part, dest)
            set_model_download_state(model_id, "completed")
            print(f"Model '{model_id}' downloaded successfully.", flush=True)
            return True
        except Exception as e:
            set_model_download_state(model_id, "error", str(e))
            print(f"Model download failed ({model_id}): {e}", flush=True)
            raise


def start_model_download(model_id):
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_id}")
    if model_is_installed(model_id):
        set_model_download_state(model_id, "completed")
        return False

    with MODEL_DOWNLOAD_STATE_LOCK:
        if MODEL_DOWNLOAD_STATE[model_id]["state"] in ("downloading", "verifying"):
            return False
        MODEL_DOWNLOAD_STATE[model_id]["state"] = "downloading"
        MODEL_DOWNLOAD_STATE[model_id]["error"] = None

    def _worker():
        try:
            download_model_sync(model_id)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()
    return True


def model_status(model_id):
    spec = MODEL_SPECS[model_id]
    installed = model_is_installed(model_id)
    with MODEL_DOWNLOAD_STATE_LOCK:
        state = MODEL_DOWNLOAD_STATE[model_id]["state"]
        error = MODEL_DOWNLOAD_STATE[model_id]["error"]

    if installed:
        state = "completed"
        downloaded = spec["size"]
    else:
        if state == "completed":
            state = "idle"
        downloaded = model_downloaded_bytes(model_id)

    percent = (downloaded / spec["size"] * 100.0) if spec["size"] else 0.0
    return {
        "id": model_id,
        "label": spec["label"],
        "description": spec["description"],
        "backend": spec.get("backend", "whisper"),
        "size": spec["size"],
        "downloaded": downloaded,
        "percent": round(min(percent, 100.0), 1),
        "installed": installed,
        "state": state,
        "error": error,
        "loaded": model is not None and loaded_model_id == model_id,
    }


def models_payload():
    return {
        "default_model": DEFAULT_MODEL,
        "active_model": loaded_model_id,
        "model_loading": MODEL_LOADING,
        "model_load_error": MODEL_LOAD_ERROR,
        "models": [model_status(model_id) for model_id in MODEL_SPECS],
    }


FFMPEG_BIN = None
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(FFMPEG_BIN)
    # imageio-ffmpeg باینری را با اسم نسخه‌دار می‌دهد؛ whisper دنبال `ffmpeg` می‌گردد.
    # اگر ffmpeg.exe کنارش نیست، یک کپی با اسم درست می‌سازیم.
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_exe) and os.path.exists(FFMPEG_BIN):
        try:
            import shutil
            shutil.copyfile(FFMPEG_BIN, ffmpeg_exe)
            print(f"Created ffmpeg.exe shim at: {ffmpeg_exe}", flush=True)
        except Exception as copy_err:
            print(f"Warning: could not create ffmpeg.exe shim: {copy_err}", flush=True)
    if ffmpeg_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"FFmpeg binary located at: {FFMPEG_BIN}", flush=True)
except Exception as e:
    print(f"Warning: imageio-ffmpeg not available: {e}", flush=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'oga', 'flac', 'aac', 'wma', 'opus', 'aif', 'aiff'}
VIDEO_EXTENSIONS = {'mp4', 'm4v', 'mkv', 'mov', 'webm', 'avi', 'mpg', 'mpeg', 'm2ts', 'ts', '3gp', 'flv'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
loaded_model_id = None
USE_GPU = torch.cuda.is_available()

# وضعیت بارگذاری مدل (برای اینکه سرور بلافاصله بالا بیاید)
MODEL_LOADING = False
MODEL_LOAD_ERROR = None
MODEL_LOAD_LOCK = threading.RLock()
# سوییچ مدل و رونویسی را سریال می‌کند تا مدل وسط کار عوض نشود.
INFERENCE_LOCK = threading.Lock()


def get_device():
    if USE_GPU:
        return "cuda"
    return "cpu"


_PUNCT_ONLY = re.compile(r"^[،؛؟!.,:;…»«()\[\]{}ٔ٬]+$")
_SENTENCE_END = re.compile(r"[.!?؟…]$")
_CLAUSE_END = re.compile(r"[،؛,;:]$")


def _split_audio_for_asr(audio, sample_rate=16000, max_seconds=26.0, min_seconds=14.0):
    """تقسیم صوت در نزدیک‌ترین سکوت به مرز ۲۶ ثانیه.

    مدل‌های Whisper فقط ۳۰ ثانیه را در یک پنجره می‌بینند. به‌جای پنجره‌های
    ثابت ۳۰ ثانیه‌ای (که کلمه را از وسط می‌برد)، نقطهٔ کم‌انرژی را انتخاب
    می‌کنیم تا هر تکه با یک وقفهٔ طبیعی تمام شود.
    """
    audio = np.asarray(audio, dtype=np.float32)
    total = int(len(audio))
    max_len = int(max_seconds * sample_rate)
    if total <= max_len:
        return [(0, total)]

    frame = max(1, int(sample_rate * 0.02))
    min_len = int(min_seconds * sample_rate)
    chunks = []
    start = 0
    while start < total:
        latest = min(start + max_len, total)
        if latest >= total:
            chunks.append((start, total))
            break

        earliest = start + min_len
        if earliest >= latest - frame:
            cut = latest
        else:
            count = (latest - earliest) // frame
            if count <= 0:
                cut = latest
            else:
                frames = audio[earliest:earliest + count * frame].reshape(count, frame)
                rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
                cut = earliest + int(np.argmin(rms)) * frame

        if cut <= start + frame:
            cut = latest
        chunks.append((start, cut))
        start = cut

    return chunks


def _join_words(words):
    text = " ".join(w["word"] for w in words)
    text = re.sub(r"\s+([،؛؟!.,:;…»\)\]\}])", r"\1", text)
    text = re.sub(r"([«\(\[])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_hf_words(raw_words, total_duration):
    """یکسان‌سازی کلمات مدل HF: ادغام نشانه‌گذاری‌های تنها و مرتب‌کردن زمان‌ها."""
    words = []
    for item in raw_words:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        try:
            if item.get("start") is None:
                start = words[-1]["end"] if words else 0.0
            else:
                start = float(item["start"])
            if item.get("end") is None:
                end = start + 0.05
            else:
                end = float(item["end"])
        except (TypeError, ValueError):
            continue

        start = max(0.0, min(start, total_duration))
        end = max(start + 0.05, min(end, total_duration))

        if _PUNCT_ONLY.match(text):
            if words:
                words[-1]["word"] += text
                words[-1]["end"] = max(words[-1]["end"], end)
            continue

        words.append({"word": text, "start": start, "end": end})

    prev_end = 0.0
    for word in words:
        if word["start"] < prev_end - 0.02:
            word["start"] = prev_end
        if word["end"] <= word["start"]:
            word["end"] = word["start"] + 0.05
        prev_end = word["end"]

    # حذف تکرارهای دقیق در مرز تکه‌ها (مثل «بعد، بعد» با فاصلهٔ صفر).
    cleaned = []
    for word in words:
        if (
            cleaned
            and word["word"] == cleaned[-1]["word"]
            and word["start"] - cleaned[-1]["end"] < 0.08
        ):
            cleaned[-1]["end"] = max(cleaned[-1]["end"], word["end"])
            continue
        cleaned.append(word)
    return cleaned


def _group_words_into_segments(words, max_chars=46, max_duration=7.0, max_gap=0.7):
    """گروه‌بندی کلمات به کیوهای کوتاه و خوانا برای SRT جمله‌ای."""
    segments = []
    current = []

    def flush():
        if not current:
            return
        text = _join_words(current)
        if text:
            start = max(0.0, current[0]["start"])
            end = max(start + 0.05, current[-1]["end"])
            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "words": [dict(w) for w in current],
            })
        current.clear()

    for word in words:
        if current:
            gap = word["start"] - current[-1]["end"]
            duration = word["end"] - current[0]["start"]
            text_len = len(_join_words(current + [word]))
            sentence_end = bool(_SENTENCE_END.search(current[-1]["word"]))
            clause_end = bool(_CLAUSE_END.search(current[-1]["word"]))
            if (
                gap > max_gap
                or duration > max_duration
                or text_len > max_chars
                or (sentence_end and len(current) >= 3)
                or (clause_end and duration > 4.0 and len(current) >= 5)
            ):
                flush()
        current.append(word)

    flush()
    return segments


class TransformersWhisperModel:
    """بک‌اند transformers برای مدل‌های Whisper فارسی (Hugging Face)."""

    def __init__(self, model_dir, device):
        self.model_dir = model_dir
        self.device = device
        self._load()

    def _load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        _prepare_hf_model_dir(self.model_dir)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            use_safetensors=True,
            attn_implementation="eager",
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()
        if getattr(model.generation_config, "forced_decoder_ids", None):
            model.generation_config.forced_decoder_ids = None

        self.processor = processor
        self.model = model
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=self.device,
        )

    def transcribe(self, filepath, language=None, word_timestamps=False):
        """رونویسی فارسی با تکه‌بندی روی سکوت و word timestamps.

        برای هر دو حالت «جمله‌ای» و «کلمه‌به‌کلمه» یک بار word timestamps
        می‌گیریم؛ حالت جمله‌ای از همان کلمات، کیوهای کوتاه می‌سازد. این کار
        از پنجره‌های ثابت ۳۰ ثانیه‌ای تمیزتر است و از بریدن کلمه جلوگیری می‌کند.
        """
        audio = whisper.audio.load_audio(filepath)
        sr = 16000
        duration = len(audio) / float(sr)
        spans = _split_audio_for_asr(audio, sr)

        raw_words = []
        for start, end in spans:
            chunk = audio[start:end]
            if len(chunk) < int(sr * 0.2):
                continue
            result = self.pipe(
                {"raw": chunk, "sampling_rate": sr},
                return_timestamps="word",
                generate_kwargs={"language": "fa", "task": "transcribe"},
            )
            offset = start / float(sr)
            chunks = result.get("chunks") or []
            if not chunks:
                text = (result.get("text") or "").strip()
                if text:
                    raw_words.append({
                        "text": text,
                        "start": offset,
                        "end": offset + len(chunk) / float(sr),
                    })
                continue
            for item in chunks:
                timestamp = item.get("timestamp") or (None, None)
                word_start, word_end = timestamp
                raw_words.append({
                    "text": item.get("text", ""),
                    "start": None if word_start is None else offset + float(word_start),
                    "end": None if word_end is None else offset + float(word_end),
                })

        words = _clean_hf_words(raw_words, duration)
        segments = _group_words_into_segments(words)
        if not segments and words:
            segments = [{
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": _join_words(words),
                "words": words,
            }]

        return {
            "language": "fa",
            "text": " ".join(segment["text"] for segment in segments).strip(),
            "segments": segments,
        }


def get_model(model_id=DEFAULT_MODEL):
    global model, loaded_model_id, MODEL_LOADING, MODEL_LOAD_ERROR

    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_id}")

    with MODEL_LOAD_LOCK:
        if model is not None and loaded_model_id == model_id:
            return model

        if not model_is_installed(model_id):
            if model_id == DEFAULT_MODEL:
                download_model_sync(model_id)
            else:
                raise RuntimeError(f"Model '{model_id}' has not been downloaded yet.")

        device = get_device()
        MODEL_LOADING = True
        MODEL_LOAD_ERROR = None
        try:
            if model is not None:
                print(f"Unloading Whisper '{loaded_model_id}' to free memory...", flush=True)
                model = None
                loaded_model_id = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(
                f"Loading model '{model_id}' ({MODEL_SPECS[model_id]['label']}) on {device}... "
                f"(1-2 min, please wait, do NOT close this window)",
                flush=True,
            )
            if MODEL_SPECS[model_id].get("backend") == "transformers":
                model = TransformersWhisperModel(MODEL_SPECS[model_id]["path"], device)
            else:
                # download_root را صریح می‌دهیم تا دقیقاً همان فایلی که دانلود کردیم استفاده شود
                model = whisper.load_model(model_id, device=device, download_root=MODEL_CACHE_DIR)
            loaded_model_id = model_id
            print(f"Model '{model_id}' loaded successfully on {device}!", flush=True)
            return model
        except Exception as e:
            MODEL_LOAD_ERROR = str(e)
            model = None
            loaded_model_id = None
            raise
        finally:
            MODEL_LOADING = False


def transcribe_with_model(model_obj, model_id, filepath, language, word_timestamps):
    """اجرای رونویسی با بک‌اند مناسب همان مدل."""
    spec = MODEL_SPECS[model_id]
    if spec.get("backend") == "transformers":
        return model_obj.transcribe(
            filepath,
            word_timestamps=word_timestamps,
        )
    return model_obj.transcribe(
        filepath,
        language=language,
        fp16=False,
        verbose=False,
        word_timestamps=word_timestamps,
    )


def preload_model_background():
    """دانلود + لود مدل پیش‌فرض در پس‌زمینه تا سرور سریع گوش بدهد."""
    global MODEL_LOAD_ERROR
    if model is not None:
        return
    try:
        with INFERENCE_LOCK:
            if not model_is_installed(DEFAULT_MODEL):
                download_model_sync(DEFAULT_MODEL)
            get_model(DEFAULT_MODEL)
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        print(f"Background model load failed: {e}", flush=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cues_to_srt(cues):
    """ساخت SRT از لیستی از (شروع، پایان، متن)."""
    srt_lines = []
    index = 0
    for start_s, end_s, text in cues:
        if not text:
            continue
        index += 1
        srt_lines.append(f"{index}")
        srt_lines.append(f"{format_timestamp(start_s)} --> {format_timestamp(end_s)}")
        srt_lines.append(text)
        srt_lines.append("")
    return "\n".join(srt_lines)


def segments_to_srt(segments):
    """زیرنویس معمولی؛ هر سگمنت Whisper یک دیالوگ مستقل."""
    cues = []
    for segment in segments:
        text = (segment.get('text') or '').strip()
        if text:
            cues.append((segment['start'], segment['end'], text))
    return cues_to_srt(cues)


def words_to_srt(segments):
    """زیرنویس کلمه‌به‌کلمه با همان زمان‌بندی اصلی Whisper.

    Whisper با word_timestamps=True برای هر کلمه start/end می‌دهد.
    هر کلمه یک کیوی مستقل SRT می‌شود تا در نرم‌افزارهای ادیت
    بتوان روی زمان‌بندی تک‌تک کلمات کار کرد.
    """
    cues = []
    for segment in segments:
        seg_start = float(segment.get('start', 0.0))
        for word in segment.get('words') or []:
            text = (word.get('word') or '').strip()
            if not text:
                continue
            try:
                start = max(0.0, float(word.get('start', seg_start)))
                end = float(word.get('end', start))
            except (TypeError, ValueError):
                continue
            # بعضی کلمات (مثل نشانه‌گذاری) طول صفر یا منفی می‌گیرند؛
            # یک حداقل کوچک می‌دهیم تا کیوی معتبر بماند.
            if end <= start:
                end = start + 0.05
            cues.append((start, end, text))
    return cues_to_srt(cues)


def count_words(segments):
    total = 0
    for segment in segments:
        total += sum(
            1 for word in (segment.get('words') or [])
            if (word.get('word') or '').strip()
        )
    return total


def make_srt_filename(suffix=''):
    """نام ذخیره‌سازی امن و یکتا برای فایل SRT."""
    return f"{uuid.uuid4().hex}{suffix}.srt"


def save_srt(content):
    name = make_srt_filename()
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return name


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """برای run.bat و فرانت‌اند: آیا سرور زنده است و مدل آماده است؟"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'active_model': loaded_model_id,
        'model_loading': MODEL_LOADING,
        'model_error': MODEL_LOAD_ERROR,
        'device': get_device(),
    })


@app.route('/models')
def list_models():
    """فهرست مدل‌ها، وضعیت نصب/دانلود و مدل فعال."""
    return jsonify(models_payload())


@app.route('/models/download', methods=['POST'])
def download_model():
    """شروع دانلود یک مدل در پس‌زمینه (دانلود توسط خود برنامه)."""
    data = request.get_json(silent=True) or request.form
    model_id = (data.get('model_id') or '').strip()
    if not model_id:
        return jsonify({'error': 'model_id is required'}), 400
    if model_id not in MODEL_SPECS:
        return jsonify({'error': f'Unknown model: {model_id}'}), 400

    try:
        started = start_model_download(model_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'success': True,
        'started': started,
        'message': 'Download started' if started else 'Model is already installed or downloading',
        **models_payload(),
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500 MB.'}), 413


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        supported = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({'error': f'File type not supported. Supported formats: {supported}'}), 400

    language = request.form.get('language', 'auto')
    output_mode = request.form.get('output_mode', 'segment').lower()
    if output_mode not in ('segment', 'word'):
        output_mode = 'segment'

    model_id = (request.form.get('model') or DEFAULT_MODEL).strip()
    if model_id not in MODEL_SPECS:
        return jsonify({'error': f'Unknown model: {model_id}'}), 400
    if not model_is_installed(model_id):
        if model_id == DEFAULT_MODEL:
            message = (
                f'The default model "{model_id}" is still downloading. '
                f'Wait for the download to finish and try again.'
            )
        else:
            message = (
                f'Model "{model_id}" is not downloaded yet. '
                f'Download it from the model section first.'
            )
        return jsonify({'error': message}), 400

    # نام اصلی را برای نمایش/دانلود نگه می‌داریم، ولی نام فایل روی دیسک
    # همیشه ASCII و امن است (نام‌های فارسی با secure_filename خالی می‌شدند).
    original_filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    ext = original_filename.rsplit('.', 1)[1].lower()
    display_base = os.path.splitext(original_filename)[0].strip() or 'media'
    safe_base = secure_filename(display_base)[:80] or 'media'
    unique_name = f"{uuid.uuid4().hex}_{safe_base}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        detect_language = None
        if language != 'auto':
            detect_language = language

        # فقط وقتی خروجی کلمه‌به‌کلمه خواسته شده، همگام‌سازی کلمات را فعال
        # می‌کنیم تا حالت معمولی مثل قبل سریع بماند.
        # قفل باعث می‌شود وقتی مدل عوض می‌شود، رونویسی دیگری وسط کار نباشد.
        with INFERENCE_LOCK:
            transcribe_model = get_model(model_id)
            result = transcribe_with_model(
                transcribe_model,
                model_id,
                filepath,
                detect_language,
                output_mode == 'word',
            )

        segments = result.get('segments', [])
        srt_content = segments_to_srt(segments)

        if output_mode == 'word':
            word_srt_content = words_to_srt(segments)
            words_count = count_words(segments)
            # اگر مدل برای فایلی هیچ کلمه‌ای برنگرداند، حداقل خروجی خالی ندهیم.
            if not word_srt_content.strip():
                word_srt_content = srt_content
            selected_content = word_srt_content
            selected_name = f"{display_base}_words.srt"
        else:
            words_count = 0
            selected_content = srt_content
            selected_name = f"{display_base}.srt"

        srt_filename = save_srt(selected_content)
        segment_srt_filename = None
        if output_mode == 'word':
            segment_srt_filename = save_srt(srt_content)

        os.remove(filepath)

        payload = {
            'success': True,
            'mode': output_mode,
            'model': model_id,
            'model_label': MODEL_SPECS[model_id]['label'],
            'srt_filename': srt_filename,
            'original_filename': selected_name,
            'detected_language': result.get('language', 'unknown'),
            'segments_count': len(segments),
            'words_count': words_count,
            'preview': selected_content[:1000],
        }
        if segment_srt_filename:
            payload['segment_srt_filename'] = segment_srt_filename
            payload['segment_original_filename'] = f"{display_base}.srt"

        return jsonify(payload)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    # جلوگیری از Path Traversal (مثل ../../windows/...)
    safe = secure_filename(filename)
    if safe != filename or '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    # نام نمایشی (مثلاً نام فارسی اصلی) فقط برای Content-Disposition است
    # و هیچ نقشی در مسیر فایل ندارد.
    download_name = request.args.get('name', '').strip()
    if download_name:
        download_name = (
            download_name
            .replace('/', '_')
            .replace('\\', '_')
            .replace('\r', '')
            .replace('\n', '')
        )[:200]
    return send_file(filepath, as_attachment=True, download_name=download_name or safe)


def cleanup_old_uploads(max_age_hours=24):
    """پاک‌سازی فایل‌های موقت قدیمی (مثل mp3های یتیم از اجراهای نیمه‌کاره).

    خروجی‌های SRT کاربر هرگز حذف نمی‌شوند تا لینک دانلودشان بعد از ۲۴ ساعت
    هم کار کند.
    """
    try:
        now = time.time()
        removed = 0
        for name in os.listdir(app.config['UPLOAD_FOLDER']):
            if name.startswith('.') or name.lower().endswith('.srt'):
                continue
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            try:
                if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_hours * 3600:
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
        if removed:
            print(f"Cleaned up {removed} old file(s) from uploads.", flush=True)
    except Exception as e:
        print(f"Upload cleanup skipped: {e}", flush=True)


if __name__ == '__main__':
    import threading

    # --- لاگ همزمان در کنسول + فایل (تا run.bat هم زنده بماند هم لاگ داشته باشد) ---
    LOGFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher.log")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    try:
        _log_fh = open(LOGFILE, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, _log_fh)
        sys.stderr = _Tee(sys.stderr, _log_fh)
    except Exception as _log_err:
        print(f"(log file disabled: {_log_err})", flush=True)

    def _maybe_auto_open_browser():
        """اگر run.bat اجرا شده باشد، وقتی سرور واقعاً بالا آمد مرورگر را باز کن."""
        if os.environ.get("AUTO_OPEN_BROWSER") != "1":
            return

        import socket
        import webbrowser

        def _wait_and_open():
            for _ in range(90):  # حدود ۳ دقیقه صبر
                try:
                    s = socket.create_connection(("127.0.0.1", 8000), timeout=1)
                    s.close()
                    time.sleep(0.5)
                    webbrowser.open("http://127.0.0.1:8000")
                    print("Browser opened automatically.", flush=True)
                    return
                except OSError:
                    time.sleep(2)
            print("Auto-open timed out; open http://127.0.0.1:8000 manually.", flush=True)

        threading.Thread(target=_wait_and_open, daemon=True).start()

    device = get_device()
    print(f"Device: {device}", flush=True)
    print("Starting Audio to SRT Converter...", flush=True)
    cleanup_old_uploads()
    # مدل پیش‌فرض را در پس‌زمینه لود می‌کنیم تا سرور بلافاصله روی پورت 8000
    # بالا بیاید. مدل High فقط وقتی کاربر از صفحه دانلودش کند نصب می‌شود.
    for _mid, _spec in MODEL_SPECS.items():
        _state = "installed" if model_is_installed(_mid) else "not downloaded"
        print(f"Model '{_mid}' ({_spec['label']}): {_state}", flush=True)
    if not model_is_installed(DEFAULT_MODEL):
        print("Default model incomplete/missing — it will be (re)downloaded in background.", flush=True)
    else:
        _default_path = MODEL_SPECS[DEFAULT_MODEL]['path']
        print(
            f"Model file found ({os.path.getsize(_default_path)} bytes). "
            f"Verifying/loading in background...",
            flush=True,
        )
    t = threading.Thread(target=preload_model_background, daemon=True)
    t.start()
    print("Server starting on http://localhost:8000 ...", flush=True)
    print("(The AI model keeps loading in the background; the page shows its status.)", flush=True)
    _maybe_auto_open_browser()
    try:
        app.run(host='0.0.0.0', port=8000, use_reloader=False, threaded=True)
    except OSError as e:
        print(f"ERROR: could not start server on port 8000: {e}", flush=True)
        print("Maybe another copy is already running? Open http://127.0.0.1:8000 manually.", flush=True)
        sys.exit(1)
