FROM ubuntu:18.04
COPY ../ /exp/mnist-dataset
COPY requirements.txt /exp/requirements.txt
COPY flask_docker.py /exp/flask_docker.py
RUN rm -f /exp/models
COPY models /exp/models
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["flask","run","--host=0.0.0.0","-p","8000"]