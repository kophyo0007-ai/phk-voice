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
CLEAN_AF = "highpass=f=70,lowpass=f=11000,afftdn=nf=-25"


def ffmpeg(args, data=None):
    return subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args],
                          input=data, capture_output=True, check=True, timeout=300).stdout


def get_ref(b64, clean=True):
    raw = base64.b64decode(b64)
    key = hashlib.sha1(raw).hexdigest()[:16] + ("c" if clean else "r")
    path = f"{REF_DIR}/{key}.wav"
    if not os.path.exists(path):
        src = f"{REF_DIR}/{key}.in"
        with open(src, "wb") as f:
            f.write(raw)
        args = ["-i", src, "-ac", "1", "-ar", "24000"]
        if clean:
            args += ["-af", CLEAN_AF]
        try:
            ffmpeg(args + [path])
        except Exception:
            ffmpeg(["-i", src, "-ac", "1", "-ar", "24000", path])
        os.remove(src)
    return path


def count_chars(t):
    return len(re.sub(r"[\s။၊,.!?]", "", t or ""))


def trim(w, thr=0.01, pad=0.08):
    idx = np.where(np.abs(w) > thr)[0]
    if len(idx) == 0:
        return w
    s = max(0, idx[0] - int(SR * pad))
    e = min(len(w), idx[-1] + int(SR * pad))
    return w[s:e]


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


def gen_part(p, kw, cps, ratios, tries):
    exp = max(0.5, count_chars(p) / cps) if cps else None
    base = float(np.median(ratios)) if len(ratios) >= 3 else 1.0
    best, best_score, best_ratio, retried = None, None, None, 0
    for attempt in range(tries):
        w = np.asarray(model.generate(text=p, **kw), dtype=np.float32)
        w = trim(w)
        if exp is None:
            return w, 0
        ratio = (len(w) / SR) / exp
        norm = ratio / base if base > 0 else ratio
        score = abs(np.log(max(norm, 1e-3)))
        if best is None or score < best_score:
            best, best_score, best_ratio = w, score, ratio
        if 0.55 <= norm <= 2.0:
            break
        retried += 1
        print(f"part retry {attempt + 1}: ratio={ratio:.2f} norm={norm:.2f}", flush=True)
    ratios.append(best_ratio)
    return best, retried


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    b64 = inp.get("ref_audio_b64")
    if not text or not b64:
        return {"error": "text and ref_audio_b64 required"}
    try:
        ref = get_ref(b64, clean=bool(inp.get("clean_ref", True)))
    except Exception as e:
        return {"error": f"ref audio convert failed: {e}"}
    reftx = (inp.get("prompt_text") or "").strip()
    cfg = float(inp.get("cfg_value", 2.0))
    steps = int(inp.get("inference_timesteps", 10))
    fmt = inp.get("format", "mp3")
    tries = max(1, min(5, int(inp.get("tries", 3))))

    parts = split_text(text)
    if not parts:
        return {"error": "empty text"}

    kw = dict(reference_wav_path=ref, cfg_value=cfg, inference_timesteps=steps,
              retry_badcase=True, retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0)
    cps = None
    if reftx:
        kw.update(prompt_wav_path=ref, prompt_text=reftx)
        try:
            d = sf.info(ref).duration
            n = count_chars(reftx)
            if d > 0 and n > 0:
                cps = min(25.0, max(6.0, n / d))
        except Exception:
            cps = None

    gap = np.zeros(int(SR * 0.25), dtype=np.float32)
    out, ratios, retried = [], [], 0
    for i, p in enumerate(parts):
        w, r = gen_part(p, kw, cps, ratios, tries)
        retried += r
        out.append(w)
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
        "retried": retried,
    }


runpod.serverless.start({"handler": handler})
