FROM nvcr.io/nvidia/pytorch:23.11-py3
RUN git config --global --add safe.directory '*'

# SWIG must already be installed for box2d-py to install correctly
RUN pip install swig

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Xvfb acts as our graphics server in Docker
RUN apt-get update && apt-get -y install xvfb

RUN pip install safety-gymnasium

RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim.appimage 
RUN chmod u+x nvim.appimage 
RUN ./nvim.appimage --appimage-extract
RUN ./squashfs-root/AppRun --version
RUN mv squashfs-root /
RUN ln -s /squashfs-root/AppRun /usr/bin/nvim

CMD ["/bin/bash"]
