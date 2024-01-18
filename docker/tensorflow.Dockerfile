FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
RUN git config --global --add safe.directory '*'

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

CMD ["/bin/bash"]