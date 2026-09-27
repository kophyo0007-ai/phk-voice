FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir voxcpm soundfile runpod

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
