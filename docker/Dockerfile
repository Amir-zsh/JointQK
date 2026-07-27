# CUDA 13 / sm90 engine image for the OSCAR-vq (vq2-longhorizon) serving fork.
#
# The base already ships torch 2.11.0+cu130, torchvision 0.26.0+cu130,
# torchaudio 2.11.0+cu130 and nvcc 13.0 — the -devel tag is kept because
# flashinfer's JIT path wants a real toolkit even though the bf16/triton
# paths never invoked nvcc in the boot smoke.
#
# sglang itself is NOT installed: serve_oscar.sh puts the fork's python/ on
# PYTHONPATH and launches `python -m sglang.launch_server`, so the image only
# needs the fork's *dependencies*.
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git numactl libnuma1 libibverbs1 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Debian marks the system interpreter externally-managed; the engine env is a
# venv with --system-site-packages so it inherits the base image's torch stack
# instead of re-downloading ~3GB of cu130 wheels.
RUN python3 -m venv --system-site-packages /opt/venv-oscar

COPY requirements.engine.cu13.txt /tmp/requirements.engine.cu13.txt
RUN /opt/venv-oscar/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv-oscar/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /tmp/requirements.engine.cu13.txt

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/venv-oscar/bin:/usr/local/cuda/bin:$PATH
# serve_oscar.sh honours this instead of $REPO_ROOT/.venv-oscar/bin/python.
ENV OSCAR_PYTHON=/opt/venv-oscar/bin/python

WORKDIR /raid/amir/quantization/teamily-project
