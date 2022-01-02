FROM pytorch/pytorch 
# I edit the nvimrc too often for it to be a base image.
# FROM nvimi
ARG USER_ID=1001
ARG GROUP_ID=101
ARG USER=app
ENV USER=$USER
ARG PROJ_ROOT=/$USER
# This is for convenience within this dockerfile. If there was a non-argument
# variable type, I'd use it instead.
ARG NEOVIM_DIR=/home/$USER/.config/nvim

RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
# When switching to mounting the whole project as a volume, it
# seemed wrong to mount it at the existing /home/app directory. So,
# going one level deeper. I think an alternative is to just work from
# /app, but I think some programs like JupyterLab have some issues
# when running from outside the home tree.
WORKDIR /home/$USER

###############################################################################
#
# Neovim
# 	
###############################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
	add-apt-repository ppa:neovim-ppa/unstable

RUN apt-get update && apt-get install -y --no-install-recommends \
	neovim  \
	git \
	locales \
	curl && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
	pip install neovim

# Set the locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

RUN conda install --yes -c conda-forge nodejs'>=12.12.0' --repodata-fn=repodata.json

###############################################################################
# 
# - Neovim
#   - ctags
#
# more info:
# 	https://github.com/universal-ctags/ctags
###############################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
	gcc \
	make \
	pkg-config \
	autoconf \ 
	automake \
	python3-docutils \
	libseccomp-dev \
	libjansson-dev \
	libyaml-dev \
	libxml2-dev && \
	rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/universal-ctags/ctags.git  && \
	cd ctags && \
	./autogen.sh && \
	./configure && \ 
	make && \
	make install && \
	cd ../  

###############################################################################
# \ctags
# \Neovim
###############################################################################

USER root
ARG PROJ_ROOT=/app

RUN mkdir $PROJ_ROOT && chown $USER $PROJ_ROOT
WORKDIR $PROJ_ROOT	

# These next two folders will be where we will mount our local data and out
# directories. We create them manually (automatic creation when mounting will
# create them as being owned by root, and then our program cannot use them).
RUN mkdir data && chown $USER data
RUN mkdir out && chown $USER out

RUN apt-get update && apt-get install -y --no-install-recommends \
	 libsm6 \
	 libxext6 \
     libxrender-dev \
     imagemagick \
	 ffmpeg && \
	 rm -rf /var/lib/apt/lists/*

###############################################################################
# LFN conda
# Source: https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
###############################################################################
ARG CONDA_ENV_NAME=light-field-networks
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
# I couldn't get this to work correctly. So hard-coding.
#RUN envname=$(head -1 /tmp/environment.yml | cut -d' ' -f2) && \
# echo "source activate " $envname > ~/.bashrc
#ENV PATH /opt/conda/envs/$envname/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" > ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH
###############################################################################
# \LFN conda
###############################################################################


RUN conda config --add channels conda-forge 
RUN conda install --yes \
# Doesn't resolve :(. Using Apt-get instead.
	flask && \
	conda clean -ya
RUN conda install -c conda-forge jupyterlab-spellchecker
#RUN jupyter labextension install jupyterlab_vim
# From: https://stackoverflow.com/questions/67050036/enable-jupyterlab-extensions-by-default-via-docker
COPY --chown=$USER proj/jupyter_notebook_config.py /etc/jupyter/
# Try to get Jupyter Lab to allow extensions on startup.
# This file was found by diffing a container running jupyterlab that had 
# extensions manually enabled.
COPY --chown=$USER proj/plugin.jupyterlab-settings /home/$USER/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension/
# Getting some permission errors printed in terminal after running Jupyter Lab, 
# and trying the below line to fix:
RUN chown -R $USER:$USER /home/$USER/.jupyter

RUN pip install --upgrade pip
RUN pip install graphviz \
        opencv-python \
		jupyterlab-vim \
        icecream \
		torchinfo \
		deprecated \
		bidict \ 
		pytest \
		ipyplot \
		htmlmin \
		xarray \
		einops \
		mypy 

# Fix permission issues with ims
# https://stackoverflow.com/a/54230833/754300
RUN rm /etc/ImageMagick-6/policy.xml



# OpenCV doesn't have pyright support. 
# Fix. https://github.com/microsoft/pylance-release/issues/138
# mypy is installed with pip above for this usecase.
RUN stubgen -m cv2 -o $(python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')

###############################################################################
# Neovim
###############################################################################
# For some reason, /home/$USER/.config is owned by root. 
RUN mkdir -p /home/$USER/.config  && chown $USER:$USER /home/$USER/.config

COPY --chown=$USER_ID proj/nvim $NEOVIM_DIR

# Currently, assume that NeoSolarized file is copied.
# RUN git clone https://github.com/overcache/NeoSolarized.git
# RUN mkdir -p $NEOVIM_DIR/colors/ && chown $USER_ID $NEOVIM_DIR/colors
# RUN cp ./NeoSolarized/colors/NeoSolarized.vim $NEOVIM_DIR/colors/

# Switching to our new user. Do this at the end, as we need root permissions 
# in order to create folders and install things.
USER $USER

RUN curl -fLo /home/$USER/.local/share/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
RUN nvim --headless +PlugInstall +qa
RUN nvim --headless -c "CocInstall -sync coc-pyright coc-html | qa"

###############################################################################
# /Neovim
###############################################################################

USER root
# In order to allow the Python package to be edited without
# a rebuild, install all code as a volume. We will still copy the
# files initially, so that things like the below pip install can work.
COPY --chown=$USER ./ ./

# Install our own project as a module.
# This is done so the tests and JupyterLab code can import it.
# RUN pip install -e ./vae

# Switching to our new user. Do this at the end, as we need root permissions 
# in order to create folders and install things.
USER $USER

