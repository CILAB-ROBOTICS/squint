FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates build-essential \
    libvulkan1 libglvnd0 libgl1 libegl1 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /workspace/squint

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# nvidia-container-toolkit mounts the NVIDIA GL/EGL/Vulkan .so libs at runtime but not
# their ICD registration jsons, so the Vulkan loader (needed by SAPIEN/ManiSkill3 for
# GPU rendering) falls back to the mesa software driver unless we provide these ourselves.
RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '{\n  "file_format_version" : "1.0.1",\n  "ICD": {\n    "library_path": "libGLX_nvidia.so.0",\n    "api_version" : "1.4.312"\n  }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json \
    && printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : {\n    "library_path" : "libEGL_nvidia.so.0"\n  }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY environment.yaml .
RUN conda env create -f environment.yaml && conda clean -afy

RUN echo "conda activate squint" >> /root/.bashrc
ENV PATH=$CONDA_DIR/envs/squint/bin:$PATH

COPY . .

CMD ["/bin/bash"]
