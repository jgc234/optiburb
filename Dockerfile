FROM ubuntu:latest

ENV TZ=Australia/Adelaide
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && apt-get install -y python3 python3-pip
RUN apt-get install -y libspatialindex-dev
RUN pip3 install shapely gpxpy numpy networkx pandas geopandas osmnx

WORKDIR /opt
