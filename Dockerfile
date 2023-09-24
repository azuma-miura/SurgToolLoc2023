# Base PyTorch image with CUDA 11.3 support
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Add a new user and switch to it
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

# Common dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 gcc g++ -y

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Python package installation
RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# mmcv-full installation compatible with CUDA 11.3
RUN python -m pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Copy mmdetection and process.py
COPY --chown=algorithm:algorithm mmdetection /opt/algorithm/mmdetection
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# Install mmdetection
WORKDIR /opt/algorithm/mmdetection
RUN pip install .

# Revert to original workdir
WORKDIR /opt/algorithm

ENTRYPOINT ["python", "-m", "process"]
