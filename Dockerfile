FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel
WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y ffmpeg git

# Python Dependencies - pin PyTorch to avoid 2.6 breaking change
RUN pip install --no-cache-dir \
    whisperx \
    pyannote.audio \
    runpod

# Force PyTorch back to 2.5.1 (whisperx upgrades it to 2.6+ which breaks pyannote checkpoint loading)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Handler kopieren
COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
