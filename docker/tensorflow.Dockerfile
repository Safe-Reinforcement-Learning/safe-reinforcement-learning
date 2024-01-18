FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
RUN git config --global --add safe.directory '*'

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Xvfb acts as our graphics server in Docker
RUN apt-get update && apt-get -y install xvfb

CMD ["/bin/bash"]