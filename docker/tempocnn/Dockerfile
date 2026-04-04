FROM tensorflow/tensorflow:2.15.0-gpu

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pinned to the tested Essentia build we already validated against TF 2.15.
RUN python -m pip install --upgrade pip \
    && python -m pip install essentia-tensorflow==2.1b6.dev1389

RUN groupadd --system tempo \
    && useradd --system --gid tempo --create-home --home-dir /home/tempo --shell /usr/sbin/nologin tempo \
    && mkdir -p /workspace \
    && chown -R tempo:tempo /workspace /home/tempo

WORKDIR /workspace
USER tempo
