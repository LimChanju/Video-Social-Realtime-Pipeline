# -------------------------------
# Base image: Apache Spark + Python 3.10
# -------------------------------
FROM apache/spark-py:v3.4.0

# 작업 디렉토리 설정
WORKDIR /app

# pip, git 설치
USER root
RUN apt-get update && apt-get install -y python3-pip && \
    pip install --upgrade pip

# requirements.txt 복사 및 설치
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Delta, PySpark용 환경 변수 설정
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_VERSION=3.5.1
ENV DELTA_VERSION=3.2.0

# Hadoop 관련 불필요한 경고 억제
ENV HADOOP_HOME=/opt/bitnami/spark
ENV PATH=$PATH:$HADOOP_HOME/bin

# 기본 명령 (필요 시 docker-compose에서 override)
CMD ["pyspark"]
