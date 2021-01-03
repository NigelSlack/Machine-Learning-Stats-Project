FROM python:3

FROM tensorflow/tensorflow:2.1.0-py3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=getPower.py

CMD flask run --host=0.0.0.0
