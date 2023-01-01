FROM python:3.8-slim-buster

EXPOSE 8081

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY datasets/ datasets/
COPY models/ models/
COPY Recommender.py Recommender.py
COPY WebServer.py WebServer.py

CMD [ "python3", "-m" , "WebServer", "run", "--host=0.0.0.0"]