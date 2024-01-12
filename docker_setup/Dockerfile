FROM python:3.10.12 as learning_gym

WORKDIR /workdir

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD *.py .

CMD ["python", "learning_gym.py"]