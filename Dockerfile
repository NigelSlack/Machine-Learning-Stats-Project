# From GMIT HDip Data Analytics 2020; Machine Learning and Statistics
FROM python:3

# Include Tensorflow for Keras Neural Network processing
# ref https://github.com/tensorflow/tensorflow/issues/38609
FROM tensorflow/tensorflow:2.1.0-py3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use 'getPower.py' as the server
ENV FLASK_APP=getPower.py

CMD flask run --host=0.0.0.0
