import os
import sys
import time
import subprocess
import torch
import whisper
import uuid
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename

MODEL_NAME = "medium"
MODEL_URL = whisper._MODELS[MODEL_NAME]
# SHA256 رسمی مدل، جزئی از URL است (دو بخش مانده به آخر)
MODEL_SHA256 = MODEL_URL.split("/")[-2]
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
MODEL_PATH = os.path.join(MODEL_CACHE_DIR, f"{MODEL_NAME}.pt")
MODEL_TARGET_SIZE = 1528008539


def ensure_model_downloaded():
    """دانلود مدل با پشتیبانی resume در صورت ناقص بودن"""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        current = os.path.getsize(MODEL_PATH)
        if current >= MODEL_TARGET_SIZE:
            print(f"Model '{MODEL_NAME}' already downloaded ({current} bytes).")
            return True
        print(f"Model '{MODEL_NAME}' incomplete ({current} bytes). Resuming download...")
    else:
        print(f"Downloading Whisper '{MODEL_NAME}' model...")

    for attempt in range(1, 200):
        cmd = ["curl", "-s", "-L", "-C", "-", "-o", MODEL_PATH, MODEL_URL]
        subprocess.run(cmd, check=False)
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MODEL_TARGET_SIZE:
            print(f"Model '{MODEL_NAME}' downloaded successfully.")
            return True
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        print(f"Iteration {attempt}: {size} bytes so far. Retrying...")
        time.sleep(2)

    raise RuntimeError("Failed to fully download the model after many attempts.")


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

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm', 'mp4', 'aac', 'wma'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
USE_GPU = torch.cuda.is_available()


def get_device():
    if USE_GPU:
        return "cuda"
    return "cpu"


def get_model():
    global model
    if model is None:
        device = get_device()
        ensure_model_downloaded()
        print(f"Loading Whisper medium model on {device}... (1-2 min, please wait, do NOT close this window)", flush=True)
        # download_root را صریح می‌دهیم تا دقیقاً همان فایلی که دانلود کردیم استفاده شود
        model = whisper.load_model(MODEL_NAME, device=device, download_root=MODEL_CACHE_DIR)
        print(f"Model loaded successfully on {device}!", flush=True)
    return model


# وضعیت بارگذاری مدل (برای اینکه سرور بلافاصله بالا بیاید)
MODEL_LOADING = False
MODEL_LOAD_ERROR = None


def preload_model_background():
    """دانلود + لود مدل در پس‌زمینه تا سرور سریع گوش بدهد."""
    global MODEL_LOADING, MODEL_LOAD_ERROR
    if model is not None or MODEL_LOADING:
        return
    MODEL_LOADING = True
    try:
        get_model()
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        print(f"Background model load failed: {e}", flush=True)
    finally:
        MODEL_LOADING = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments):
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    return "\n".join(srt_lines)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """برای run.bat و فرانت‌اند: آیا سرور زنده است و مدل آماده است؟"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_loading': MODEL_LOADING,
        'model_error': MODEL_LOAD_ERROR,
        'device': get_device(),
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
        return jsonify({'error': 'File type not supported. Supported formats: mp3, wav, m4a, ogg, flac, webm, mp4, aac, wma'}), 400

    language = request.form.get('language', 'auto')

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        whisper_model = get_model()
        device = get_device()

        detect_language = None
        if language != 'auto':
            detect_language = language

        result = whisper_model.transcribe(
            filepath,
            language=detect_language,
            fp16=False,
            verbose=False
        )

        srt_content = segments_to_srt(result['segments'])

        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        srt_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{srt_filename}")

        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        os.remove(filepath)

        return jsonify({
            'success': True,
            'srt_filename': os.path.basename(srt_path),
            'original_filename': srt_filename,
            'detected_language': result.get('language', 'unknown'),
            'segments_count': len(result['segments']),
            'preview': srt_content[:1000]
        })

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
    return send_file(filepath, as_attachment=True)


def cleanup_old_uploads(max_age_hours=24):
    """پاک‌سازی فایل‌های موقت قدیمی (مثل mp3های یتیم از اجراهای نیمه‌کاره)."""
    try:
        now = time.time()
        removed = 0
        for name in os.listdir(app.config['UPLOAD_FOLDER']):
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
    print("Starting Speech to Text...", flush=True)
    cleanup_old_uploads()
    # مدل را در پس‌زمینه لود می‌کنیم تا سرور بلافاصله روی پورت 8000 بالا بیاید
    # و مرورگر معطل نماند. اولین درخواست /upload اگر مدل هنوز لود نشده باشد،
    # خودش صبر می‌کند تا لود تمام شود.
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < MODEL_TARGET_SIZE:
        print("Model file incomplete/missing — it will be (re)downloaded in background.", flush=True)
    else:
        print(f"Model file found ({os.path.getsize(MODEL_PATH)} bytes). Verifying/loading in background...", flush=True)
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
