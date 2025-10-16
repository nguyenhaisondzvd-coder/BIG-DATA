FROM python:3.9-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    gnupg \
    wget \
    curl \
    ssh \
    openssh-client \
    procps \
    net-tools \
    openjdk-17-jdk-headless \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Download and install Hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.4/hadoop-3.3.4.tar.gz \
    && tar -xzf hadoop-3.3.4.tar.gz -C /opt/ \
    && mv /opt/hadoop-3.3.4 /opt/hadoop \
    && rm hadoop-3.3.4.tar.gz

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz \
    && tar -xzf spark-3.4.0-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.4.0-bin-hadoop3 /opt/spark \
    && rm spark-3.4.0-bin-hadoop3.tgz

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop
ENV SPARK_HOME=/opt/spark
ENV PATH=$HADOOP_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
ENV PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV SPARK_CONF_DIR=/opt/spark/conf

# Hadoop user environment variables
ENV HDFS_NAMENODE_USER=root
ENV HDFS_DATANODE_USER=root
ENV HDFS_SECONDARYNAMENODE_USER=root
ENV YARN_RESOURCEMANAGER_USER=root
ENV YARN_NODEMANAGER_USER=root

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/results \
    /opt/hadoop/logs /opt/spark/logs \
    /opt/hadoop/dfs/name /opt/hadoop/dfs/data

# SSH setup for Hadoop
RUN mkdir -p ~/.ssh && \
    ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys

# Create Hadoop configuration directories
RUN mkdir -p /opt/hadoop/etc/hadoop

EXPOSE 8080 8081 4040 7077 9000 9870 9864 8042

CMD ["/bin/bash", "-c", "tail -f /dev/null"]