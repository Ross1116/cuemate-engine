FROM tensorflow/tensorflow:2.15.0-gpu

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip \
    && python -m pip install essentia-tensorflow

WORKDIR /workspace
