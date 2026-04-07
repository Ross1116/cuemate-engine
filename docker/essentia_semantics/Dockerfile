FROM tensorflow/tensorflow:2.15.0-gpu

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Match the tested Essentia TensorFlow build we already use elsewhere.
RUN python -m pip install --upgrade pip \
    && python -m pip install essentia-tensorflow==2.1b6.dev1389

RUN groupadd --system essentia \
    && useradd --system --gid essentia --create-home --home-dir /home/essentia --shell /usr/sbin/nologin essentia \
    && mkdir -p /workspace \
    && chown -R essentia:essentia /workspace /home/essentia

WORKDIR /workspace
USER essentia
