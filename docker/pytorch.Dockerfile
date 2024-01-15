FROM nvcr.io/nvidia/pytorch:23.11-py3

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

CMD ["/bin/bash"]