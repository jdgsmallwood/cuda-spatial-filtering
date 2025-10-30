FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

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
    libhdf5-dev \
    hdf5-tools \
    valgrind \
    iwyu \
    tmux \
    gdb \
    && rm -rf /var/lib/apt/lists/*


# Install cuTENSOR (adjust the version as desired)
RUN wget https://developer.download.nvidia.com/compute/cutensor/2.2.0/local_installers/cutensor-local-repo-ubuntu2404-2.2.0_1.0-1_amd64.deb
RUN dpkg -i cutensor-local-repo-ubuntu2404-2.2.0_1.0-1_amd64.deb
RUN cp /var/cutensor-local-repo-ubuntu2404-2.2.0/cutensor-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install libcutensor2 libcutensor-dev libcutensor-doc
   

RUN    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb
RUN    dpkg -i NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb && rm NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb


RUN    test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN    apt-get update
RUN    test -f /usr/share/doc/kitware-archive-keyring/copyright || rm /usr/share/keyrings/kitware-archive-keyring.gpg
RUN    apt-get install -y kitware-archive-keyring
RUN    apt-get install -y cmake

# Set default compiler environment variables (optional)
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV PATH=/usr/local/cuda-12.8/compute-sanitizer:$PATH

RUN echo "set-option -g prefix C-a" > /etc/tmux.conf && \
    echo "unbind-key C-b" >> /etc/tmux.conf && \
    echo "bind-key C-a send-prefix" >> /etc/tmux.conf && \
    echo "set-option -g status-bg red" >> /etc/tmux.conf

ENV     CUTENSOR_ROOT=/usr/lib/x86_64-linux-gnu/libcutensor
WORKDIR /workspace
