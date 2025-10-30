FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and add Kitware's CMake repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    wget 

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | \
    tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Install GCC, CMake, and common build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    ninja-build \
    python3-pip \
    flex \
    bison \
    linux-tools-common \ 
    linux-tools-generic \
    linux-tools-$(uname -r) \
    && rm -rf /var/lib/apt/lists/*

# Set default compiler environment variables (optional)
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64/stubs:$LD_LIBRARY_PATH

RUN ln -s /usr/local/cuda-12.9/lib64/stubs/libcuda.so /usr/local/cuda-12.9/lib64/stubs/libcuda.so.1

WORKDIR /workspace
