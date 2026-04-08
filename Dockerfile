FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /workdir \
    && chown user:user /opt/app /input /output /workdir

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

# Download model weights from HuggingFace at build time
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('LMMasters/GHMSCHRFT-v1', local_dir='/opt/app/model')"

# Run offline after build (model is already on disk)
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PYTHONUNBUFFERED=1

COPY --chown=user:user nlp-task-configuration.json /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user api.py /opt/app/
