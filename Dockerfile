# syntax=docker/dockerfile:1
FROM openjdk:8-jdk-alpine
FROM docker.io/bitnami/spark:3

USER root
WORKDIR /opt/bitnami/

RUN mkdir -p /opt/bitnami/local/files
RUN mkdir -p /opt/bitnami/local/jar
RUN mkdir -p /opt/bitnami/local/model/linearModel
RUN mkdir -p /opt/bitnami/local/model/logisticmodel


COPY TrainingDataset.csv /opt/bitnami/local/files
COPY ValidationDataset.csv /opt/bitnami/local/files


COPY linearModel /opt/bitnami/local/model/linearModel
COPY logisticmodel /opt/bitnami/local/model/logisticmodel

COPY QualityPredictionEngine-jar-with-dependencies.jar /opt/bitnami/local/jar

EXPOSE 7077
EXPOSE 8080

CMD /opt/bitnami/spark/bin/spark-submit \  
    --master spark://`hostname --fqdn`:4040\
    --class org.njit.cloudcomputing.QualityPrediction \
    /opt/bitnami/local/jar/QualityPredictionEngine-jar-with-dependencies.jar  /opt/bitnami/local/files/ValidationDataset.csv local[*] /opt/bitnami/local/model
