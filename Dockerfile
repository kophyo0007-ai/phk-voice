FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir voxcpm soundfile
RUN pip install --no-cache-dir runpod

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
