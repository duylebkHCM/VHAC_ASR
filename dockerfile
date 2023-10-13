FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version

COPY environment.yml src/environment.yml

RUN conda env create -f src/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "duyla_seqmodel", "/bin/bash", "-c"]

# RUN conda activate duyla_seqmodel

RUN pip install https://github.com/kpu/kenlm/archive/master.zip
