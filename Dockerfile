FROM nvcr.io/nvidia/pytorch:24.05-py3
LABEL maintainer "Toskov Jason <jason.toskov@epfl.ch>"

## user: root permission
USER root

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update && apt-get install -y \
  cmake \
  make\
  vim \
  htop \
  locales \
  unzip \
  wget \
  curl \
  ca-certificates \
  sudo \
  git \
  nano \
  psmisc \
  screen \
  tmux \
  gcc \
  g++-10 \
  htop \
  bzip2 \
  parallel \
  ffmpeg \
  ninja-build \
  build-essential \
  imagemagick \
  python3-dev \
  python3-venv \
  # For colmap\
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libeigen3-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libsqlite3-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  libceres-dev \
  libx11-6 \
  libssl-dev \
  libffi-dev \
  libfreetype6 \
  libgl1-mesa-dev \
  libgl1-mesa-glx \
  libglu1-mesa \
  libxi6 \
  libxrender1 \
  libsm6 \
  libxext6 \
  man-db \
  manpages-posix \
  # For flann\
  liblz4-dev \
  # For blender \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/*



# For Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | /bin/bash
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*


# Configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen


# Create user with UID=NB_UID and in the 'NB_GID' group
# and make sure these dirs are writable by the `NB_GID` group.
# Configure user and group
#################################################
# CHANGE THIS PART ACCORDINGLY !!!
# ENV SHELL=/bin/bash \
#     NB_USER= \
#     NB_UID= \
#     NB_GROUP= \
#     NB_GID=
#################################################

ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
  echo "${NB_USER}:${NB_USER}" | chpasswd && \
  usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

# user: user permission
USER $NB_USER
WORKDIR $HOME


# Create conda environment
# Install Miniconda
RUN curl -Lso ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py310_24.4.0-0-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python environment
RUN $HOME/miniconda/bin/conda create -n mast3r python=3.11 cmake=3.14.0 \
 && $HOME/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=mast3r
ENV CONDA_PREFIX=$HOME/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN $HOME/miniconda/bin/conda clean -ya

RUN conda install -y pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

RUN echo $CONDA_PREFIX

RUN pip install --upgrade pip
RUN pip install \
  scikit-learn \
  roma \
  gradio \
  matplotlib \
  tqdm \
  opencv-python \
  scipy \
  einops \
  trimesh \
  tensorboard \
  "pyglet<2" \
  "huggingface-hub[torch]>=0.22" \
  pillow-heif \
  pyrender \
  kapture \
  kapture-localization \
  numpy-quaternion \
  pycolmap \
  poselib

RUN pip install faiss-cpu asmk ipython ipykernel open3d

RUN pip install \
  diffusers \
  transformers \
  accelerate \
  bitsandbytes \
  imageio-ffmpeg

# You need to be building from within the git repo cloned locally for this to work
COPY --chown=$NB_USER . $HOME/perspective-analysis-of-generated-images/

WORKDIR $HOME/perspective-analysis-of-generated-images

RUN cp feature_extraction/pyproject_for_mast3r.toml mast3r/pyproject.toml
RUN cp feature_extraction/pyproject_for_dust3r.toml mast3r/dust3r/pyproject.toml

WORKDIR $HOME/perspective-analysis-of-generated-images/mast3r

RUN pip install -e .
RUN pip install -e ./dust3r

RUN pip install cython
RUN git clone https://github.com/jenicek/asmk \
  && cd asmk/cython/ \
  && cythonize *.pyx \
  && cd .. \
  && python3 setup.py build_ext --inplace \
  && cd ..

# ENV PATH=$HOME/miniconda/bin:$PATH

RUN echo "" >> ~/.bashrc; echo "" >> ~/.bashrc; echo "export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}" >> ~/.bashrc; echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc

