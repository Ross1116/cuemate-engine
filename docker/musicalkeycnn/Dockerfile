FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        librosa>=0.10.2 \
        numpy>=1.26 \
        scipy>=1.13 \
        soundfile>=0.12.1

WORKDIR /workspace
