FROM python:3.13-slim

RUN apt-get update
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-gdal \
    libgdal-dev \
    libspatialindex-dev \
    build-essential

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt
