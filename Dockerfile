# Core Starspin build and reviewer environment; pulsar-analysis tools are optional.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git wget build-essential ninja-build python3-pip \
    autoconf automake libtool pkg-config gfortran flex bison \
    libhdf5-dev hdf5-tools libcfitsio-dev casacore-dev casacore-data \
    libboost-dev libhiredis-dev libibverbs-dev libnuma-dev libltdl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir cmake==3.31.6
# Debian installs JPL tables outside casacore's runtime search directory.
RUN for table in DE200 DE405; do \
      ln -s /usr/share/casacore/data/ephemerides/$table /var/lib/casacore/data/ephemerides/$table; \
    done
RUN wget -4 --timeout=30 --tries=3 --progress=dot:giga https://developer.download.nvidia.com/compute/cutensor/2.2.0/local_installers/cutensor-local-repo-ubuntu2204-2.2.0_1.0-1_amd64.deb \
    && dpkg -i cutensor-local-repo-ubuntu2204-2.2.0_1.0-1_amd64.deb \
    && dpkg -i /var/cutensor-local-repo-ubuntu2204-2.2.0/libcutensor*.deb \
    && rm -rf /var/lib/apt/lists/* cutensor-local-repo-ubuntu2204-2.2.0_1.0-1_amd64.deb
RUN ln -s /usr/bin/python3 /usr/local/bin/python
RUN git clone https://git.code.sf.net/p/psrdada/code /tmp/psrdada \
    && cd /tmp/psrdada && git checkout 151198ea3ec4a3a237f51e6217a5a2d7ff39194e \
    && ./bootstrap && ./configure --prefix=/usr/local \
    && make -j2 && make install && ldconfig && rm -rf /tmp/psrdada
COPY requirements-test.txt /tmp/requirements-test.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-test.txt
RUN cp /var/cutensor-local-repo-ubuntu2204-2.2.0/cutensor-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get install -y --no-install-recommends libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PSRHOME=/usr/local CUDA_HOME=/usr/local/cuda
WORKDIR /workspace
CMD ["/bin/bash"]
