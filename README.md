# 🎙️ Speech to Text — تبدیل هوشمند صوت به زیرنویس SRT

> Flask + OpenAI Whisper — آپلود فایل صوتی/تصویری، تشخیص گفتار و دانلود زیرنویس `.srt`
> رابط کاربری فارسی و راست‌چین با فونت وزیرمتن

## ✨ ویژگی‌ها

- 🎧 آپلود فایل تا **500 مگابایت** با درگ‌دراپ + پیش‌نمایش پخش صوت
- 🌍 زبان: **تشخیص خودکار / فارسی / انگلیسی**
- 🧠 مدل **Whisper `medium`** با دانلود خودکار + پشتیبانی از **resume** (ادامه دانلود ناقص)
- ⚡ لود مدل در **پس‌زمینه** — سرور بلافاصله بالا می‌آید، صفحه وضعیت مدل را نشان می‌دهد
- 🎮 تشخیص خودکار **GPU (CUDA)** و fallback به CPU
- 📝 تولید **SRT استاندارد** + پیش‌نمایش متن + نمایش زبان شناسایی‌شده و تعداد سگمنت‌ها
- ⬇️ دانلود فایل `.srt` (محافظت‌شده در برابر Path Traversal)
- 🧹 پاک‌سازی خودکار فایل‌های موقت قدیمی‌تر از ۲۴ ساعت
- 🖥️ لانچر ویندوزی (`run.bat`) با باز شدن خودکار مرورگر و نمایش لاگ زنده
- 💜 UI تیره، مدرن و ریسپانسیو (Vazirmatn، RTL)

فرمت‌های پشتیبانی‌شده: `mp3, wav, m4a, ogg, flac, webm, mp4, aac, wma`

## 🛠️ تکنولوژی‌ها

| بخش | تکنولوژی |
|---|---|
| Backend | Python 3.11, Flask, openai-whisper, PyTorch |
| Audio | imageio-ffmpeg (باینری خودکار FFmpeg) |
| Frontend | HTML/CSS/JS تک‌فایل، Vazirmatn، بدون بیلد |
| Launcher | `run.bat` (ویندوز) |

## ✅ پیش‌نیازها

- Windows 10/11 (64-bit)
- Python **3.11**
- کارت NVIDIA + درایور CUDA (اختیاری — برای سرعت بیشتر؛ بدون آن روی CPU کار می‌کند)
- اینترنت برای دانلود اول مدل (~1.5 گیگ)

## 🚀 اجرا (پیشنهادی — ویندوز)

```bat
# 1) یک‌بار: ساخت محیط مجازی پایتون 3.11
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt

# 2) هر بار: دابل‌کلیک روی run.bat
run.bat
```

مرورگر خودکار روی http://127.0.0.1:8000 باز می‌شود.
(اولین اجرا ۱–۲ دقیقه طول می‌کشد چون مدل دانلود/لود می‌شود — پنجره را نبندید.)

## 🐍 اجرای دستی

```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
python app.py
# http://127.0.0.1:8000
```

## 🖱️ نحوه استفاده

1. فایل صوتی را درگ کنید یا کلیک کنید و انتخاب کنید
2. زبان را انتخاب کنید (تشخیص خودکار / فارسی / English)
3. دکمه **تبدیل به زیرنویس** را بزنید
4. پیش‌نمایش متن را ببینید و فایل `.srt` را دانلود کنید

## 🔌 API

| متد | مسیر | توضیح |
|---|---|---|
| `GET` | `/` | صفحه اصلی |
| `GET` | `/health` | وضعیت سرور و مدل (`model_loaded`, `model_loading`, `device`) |
| `POST` | `/upload` | آپلود فایل (`file` + فیلد `language`: `auto`/`fa`/`en`) — خروجی JSON شامل `srt_filename` و `preview` |
| `GET` | `/download/<filename>` | دانلود فایل SRT |

مثال:

```bash
curl -F "file=@voice.mp3" -F "language=auto" http://127.0.0.1:8000/upload
```

## 📁 ساختار پروژه

```
TTS/
├── app.py              # بک‌اند Flask + Whisper
├── run.bat             # لانچر ویندوز
├── requirements.txt
├── templates/
│   └── index.html      # فرانت‌اند فارسی
└── uploads/            # فایل‌های موقت (در گیت نیست)
```

## ⚙️ تنظیمات مهم

- مدل در `app.py` با `MODEL_NAME = "medium"` قابل تغییر است (`tiny/base/small/medium/large`)
- پورت: `8000` | حد آپلود: `500MB` (`MAX_CONTENT_LENGTH`)
- کش مدل: `~/.cache/whisper`

## 📌 نکات

- پوشه‌های `venv311/` ، `uploads/*` و `launcher.log` در `.gitignore` هستند و در ریپو نیستند.
- اگر پورت 8000 اشغال باشد، `run.bat` به‌جای اجرای مجدد فقط مرورگر را باز می‌کند.
