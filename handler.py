import os, io, re, base64, hashlib, subprocess
import numpy as np
import soundfile as sf
import runpod
from voxcpm import VoxCPM

print("loading model...", flush=True)
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = model.tts_model.sample_rate
print("MODEL READY", flush=True)

REF_DIR = "/tmp/refs"
os.makedirs(REF_DIR, exist_ok=True)


def ffmpeg(args, data=None):
    return subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args],
                          input=data, capture_output=True, check=True, timeout=300).stdout


def get_ref(b64):
    raw = base64.b64decode(b64)
    key = hashlib.sha1(raw).hexdigest()[:16]
    path = f"{REF_DIR}/{key}.wav"
    if not os.path.exists(path):
        src = f"{REF_DIR}/{key}.in"
        with open(src, "wb") as f:
            f.write(raw)
        ffmpeg(["-i", src, "-ac", "1", "-ar", "24000", path])
        os.remove(src)
    return path


def split_text(text):
    parts = []
    for s in re.split(r"[။\n]+", text):
        s = s.strip()
        if not s:
            continue
        if len(s) <= 250:
            parts.append(s + "။")
            continue
        chunk = ""
        for piece in (x.strip() for x in s.split("၊")):
            if not piece:
                continue
            if chunk and len(chunk) + len(piece) > 250:
                parts.append(chunk + "။")
                chunk = piece
            else:
                chunk = f"{chunk}၊ {piece}" if chunk else piece
        if chunk:
            parts.append(chunk + "။")
    return parts


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    b64 = inp.get("ref_audio_b64")
    if not text or not b64:
        return {"error": "text and ref_audio_b64 required"}
    try:
        ref = get_ref(b64)
    except Exception as e:
        return {"error": f"ref audio convert failed: {e}"}
    reftx = (inp.get("prompt_text") or "").strip()
    cfg = float(inp.get("cfg_value", 2.0))
    steps = int(inp.get("inference_timesteps", 10))
    fmt = inp.get("format", "mp3")

    parts = split_text(text)
    if not parts:
        return {"error": "empty text"}
    gap = np.zeros(int(SR * 0.3), dtype=np.float32)
    out = []
    for i, p in enumerate(parts):
        kw = dict(text=p, reference_wav_path=ref, cfg_value=cfg, inference_timesteps=steps)
        if reftx:
            kw.update(prompt_wav_path=ref, prompt_text=reftx)
        out.append(np.asarray(model.generate(**kw), dtype=np.float32))
        if i < len(parts) - 1:
            out.append(gap)
        runpod.serverless.progress_update(job, f"{i + 1}/{len(parts)}")

    wav = np.concatenate(out)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV")
    data = buf.getvalue()
    if fmt == "mp3":
        data = ffmpeg(["-f", "wav", "-i", "pipe:0", "-ac", "1", "-b:a", "96k",
                       "-f", "mp3", "pipe:1"], data)
    return {
        "audio_b64": base64.b64encode(data).decode(),
        "format": fmt,
        "sample_rate": SR,
        "duration": round(len(wav) / SR, 2),
        "parts": len(parts),
    }


runpod.serverless.start({"handler": handler})
