import os, io, re, base64, random, hashlib, subprocess
import numpy as np
import soundfile as sf
import torch
import runpod
from voxcpm import VoxCPM

print("loading model...", flush=True)
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)  # denoiser off = no muffled prompt enhancement
SR = model.tts_model.sample_rate
print(f"MODEL READY (sample_rate={SR})", flush=True)

REF_DIR = "/tmp/refs"
os.makedirs(REF_DIR, exist_ok=True)
REF_SR = 48000  # keep reference detail; VoxCPM resamples internally as needed
CLEAN_AF = "highpass=f=70,afftdn=nf=-25"  # removed lowpass=11000 (it cut clarity)

# natural breathing pauses (seconds) - can be overridden per request
PAUSE_SENT_MIN, PAUSE_SENT_MAX = 0.30, 0.45
PAUSE_COMMA_MIN, PAUSE_COMMA_MAX = 0.18, 0.26


def ffmpeg(args, data=None):
    return subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args],
                          input=data, capture_output=True, check=True, timeout=300).stdout


def get_ref(b64, clean=True):
    raw = base64.b64decode(b64)
    key = hashlib.sha1(raw).hexdigest()[:16] + ("c" if clean else "r") + str(REF_SR)
    path = f"{REF_DIR}/{key}.wav"
    if not os.path.exists(path):
        src = f"{REF_DIR}/{key}.in"
        with open(src, "wb") as f:
            f.write(raw)
        args = ["-i", src, "-ac", "1", "-ar", str(REF_SR)]
        if clean:
            args += ["-af", CLEAN_AF]
        try:
            ffmpeg(args + ["-c:a", "pcm_s16le", path])
        except Exception:
            ffmpeg(["-i", src, "-ac", "1", "-ar", str(REF_SR), "-c:a", "pcm_s16le", path])
        os.remove(src)
    return path, int(hashlib.sha1(raw).hexdigest()[:8], 16)


def set_seed(s):
    s = int(s) % (2 ** 31)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def count_chars(t):
    return len(re.sub(r"[\s။၊,.!?]", "", t or ""))


def trim(w, thr=0.01, pad=0.10):
    # pad 0.06 -> 0.10 so soft word endings / breaths are not clipped
    idx = np.where(np.abs(w) > thr)[0]
    if len(idx) == 0:
        return w
    s = max(0, idx[0] - int(SR * pad))
    e = min(len(w), idx[-1] + int(SR * pad))
    w = w[s:e].copy()
    # 10ms fade in/out to avoid clicks at joins
    f = min(len(w) // 4, int(SR * 0.01))
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        w[:f] *= ramp
        w[-f:] *= ramp[::-1]
    return w


def active_rms(w, thr=0.01):
    a = w[np.abs(w) > thr]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0


def end_mark(s):
    return s if s[-1:] in ".?!" else s + "။"


def split_text(text):
    # Sentence split on Burmese ။ and English . ? ! ; long sentences further split on ၊ and ,
    # Returns (part, kind) where kind is "sent" or "comma" (pause after the part)
    text = re.sub(r"([.?!])(\s+|$)", r"\1။", text)
    parts = []
    for s in re.split(r"[။\n]+", text):
        s = s.strip()
        if not s:
            continue
        if len(s) <= 250:
            parts.append((end_mark(s), "sent"))
            continue
        pieces = [x.strip() for x in re.split(r"[၊,]", s) if x.strip()]
        chunk = ""
        for piece in pieces:
            if chunk and len(chunk) + len(piece) > 250:
                parts.append((end_mark(chunk), "comma"))
                chunk = piece
            else:
                chunk = f"{chunk}၊ {piece}" if chunk else piece
        if chunk:
            parts.append((end_mark(chunk), "sent"))
    return parts


def gen_part(p, kw, seed, cps, ratios, tries, style=""):
    exp = max(0.5, count_chars(p) / cps) if cps else None
    base = float(np.median(ratios)) if len(ratios) >= 3 else 1.0
    text_in = f"({style}){p}" if style else p
    best, best_score, best_ratio, retried = None, None, None, 0
    for attempt in range(tries):
        set_seed(seed + attempt * 7919)
        w = trim(np.asarray(model.generate(text=text_in, **kw), dtype=np.float32))
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


def match_loudness(waves):
    rms = [active_rms(w) for w in waves]
    valid = [r for r in rms if r > 0]
    if not valid:
        return waves
    target = float(np.median(valid))
    out = []
    for w, r in zip(waves, rms):
        if r > 0:
            g = max(10 ** (-3 / 20), min(10 ** (3 / 20), target / r))
            peak = float(np.max(np.abs(w))) or 1.0
            g = min(g, 0.98 / peak)
            w = w * g
        out.append(w)
    return out


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    b64 = inp.get("ref_audio_b64")
    if not text or not b64:
        return {"error": "text and ref_audio_b64 required"}
    try:
        ref, ref_seed = get_ref(b64, clean=bool(inp.get("clean_ref", True)))
    except Exception as e:
        return {"error": f"ref audio convert failed: {e}"}
    reftx = (inp.get("prompt_text") or "").strip()
    mode = inp.get("clone_mode", "hifi")  # default hifi: uses prompt_text for closer voice match
    cfg = float(inp.get("cfg_value", 2.0))
    steps = int(inp.get("inference_timesteps", 10))
    fmt = inp.get("format", "wav")
    tries = max(1, min(5, int(inp.get("tries", 3))))
    seed = int(inp.get("seed", ref_seed))
    style = re.sub(r"[()]", "", (inp.get("style") or "")).strip()[:200]
    ps_min = float(inp.get("pause_min", PAUSE_SENT_MIN))
    ps_max = float(inp.get("pause_max", PAUSE_SENT_MAX))

    parts = split_text(text)
    if not parts:
        return {"error": "empty text"}

    kw = dict(reference_wav_path=ref, cfg_value=cfg, inference_timesteps=steps,
              retry_badcase=True, retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0)
    if mode == "hifi" and reftx:
        kw.update(prompt_wav_path=ref, prompt_text=reftx)

    cps = None
    if reftx:
        try:
            d = sf.info(ref).duration
            n = count_chars(reftx)
            if d > 0 and n > 0:
                cps = min(25.0, max(6.0, n / d))
        except Exception:
            cps = None

    rng = random.Random(seed)
    waves, pauses, ratios, retried = [], [], [], 0
    for i, (p, kind) in enumerate(parts):
        try:
            w, r = gen_part(p, kw, seed, cps, ratios, tries, style)
        except Exception as e:
            if "prompt_wav_path" in kw:
                print(f"hifi failed, fallback to ref-only: {e}", flush=True)
                kw.pop("prompt_wav_path", None)
                kw.pop("prompt_text", None)
                w, r = gen_part(p, kw, seed, cps, ratios, tries, style)
            else:
                raise
        retried += r
        waves.append(w)
        if kind == "sent":
            pauses.append(rng.uniform(ps_min, ps_max))
        else:
            pauses.append(rng.uniform(PAUSE_COMMA_MIN, PAUSE_COMMA_MAX))
        runpod.serverless.progress_update(job, f"{i + 1}/{len(parts)}")

    waves = match_loudness(waves)
    out = []
    for i, w in enumerate(waves):
        out.append(w)
        if i < len(waves) - 1:
            out.append(np.zeros(int(SR * pauses[i]), dtype=np.float32))
    wav = np.concatenate(out)

    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    data = buf.getvalue()
    if fmt == "mp3":
        data = ffmpeg(["-f", "wav", "-i", "pipe:0", "-ac", "1", "-b:a", "192k",
                       "-f", "mp3", "pipe:1"], data)
    return {
        "audio_b64": base64.b64encode(data).decode(),
        "format": fmt,
        "sample_rate": SR,
        "duration": round(len(wav) / SR, 2),
        "parts": len(parts),
        "retried": retried,
        "mode": "hifi" if "prompt_wav_path" in kw else "ref",
        "seed": seed,
        "style": style,
    }


runpod.serverless.start({"handler": handler})
