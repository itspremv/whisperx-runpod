FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y ffmpeg git

# Python Dependencies
RUN pip install --no-cache-dir \
    whisperx \
    pyannote.audio \
    runpod

# Pin PyTorch to 2.5.1 to avoid the weights_only breaking change in 2.6
RUN pip install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Handler kopieren
COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
