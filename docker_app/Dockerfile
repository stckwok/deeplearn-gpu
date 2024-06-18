FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

LABEL maintainer="Sam K"

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

# Install basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git tar wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

# Install mamba
RUN conda install -y mamba -c conda-forge

ADD ./environment.yml .
RUN mamba env update --file ./environment.yml &&\
    conda clean -tipy

USER root
RUN git clone --depth 1 https://github.com/stckwok/deeplearn-gpu.git /home/$USERNAME/deeplearn-gpu

# For interactive shell
RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc
