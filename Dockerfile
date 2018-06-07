FROM nvcr.io/nvidia/tensorflow:18.05

RUN apt-get update && apt-get install --no-install-recommends -y \
        git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.11.3 \
  scipy \
  networkx==1.10 \
  scikit-learn

# Installs Java.
ENV JAVA_VER 8
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle
RUN echo 'deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
    echo 'deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
    apt-get update && \
    echo oracle-java${JAVA_VER}-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections && \
    apt-get install -y --force-yes --no-install-recommends oracle-java${JAVA_VER}-installer oracle-java${JAVA_VER}-set-default && \
    apt-get clean && \
    rm -rf /var/cache/oracle-jdk${JAVA_VER}-installer
RUN update-java-alternatives -s java-8-oracle
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-oracle" >> ~/.bashrc
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installs Ant.
ENV ANT_VERSION 1.10.1
RUN cd && \
    wget -q http://www.us.apache.org/dist//ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz && \
    tar -xzf apache-ant-${ANT_VERSION}-bin.tar.gz && \
    mv apache-ant-${ANT_VERSION} /opt/ant && \
    rm apache-ant-${ANT_VERSION}-bin.tar.gz
ENV ANT_HOME /opt/ant
ENV PATH ${PATH}:/opt/ant/bin


WORKDIR /GraphEmbedding

# rsync -avzh ~/Documents/GraphEmbedding yba@qilin.cs.ucla.edu:/home/yba/
# docker build . -t yba_graphembedding
# nvidia-docker run -v /home/yba/yba_GraphEmbedding:/yba_GraphEmbedding --env CUDA_VISIBLE_DEVICES=_ yba_GraphEmbedding
