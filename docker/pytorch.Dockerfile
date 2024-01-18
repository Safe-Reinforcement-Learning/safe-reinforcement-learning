FROM nvcr.io/nvidia/pytorch:23.11-py3
RUN git config --global --add safe.directory '*'

# SWIG must already be installed for box2d-py to install correctly
RUN pip install swig

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Xvfb acts as our graphics server in Docker
RUN apt-get update && apt-get -y install xvfb

CMD ["/bin/bash"]