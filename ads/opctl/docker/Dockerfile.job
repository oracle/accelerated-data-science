# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

FROM  --platform=linux/amd64 ghcr.io/oracle/oraclelinux:7-slim

# Configure environment
ENV DATASCIENCE_USER datascience
ENV DATASCIENCE_UID 1000
ENV HOME /home/$DATASCIENCE_USER
ENV DATASCIENCE_INSTALL_DIR /etc/datascience
ENV LOGS_DIRECTORY /logs

RUN \
    yum -y -q install oracle-release-el7 deltarpm && \
    yum-config-manager --add-repo http://yum.oracle.com/repo/OracleLinux/OL7/developer_EPEL/x86_64 && \
    yum-config-manager --enable ol7_optional_latest --enable ol7_addons --enable ol7_software_collections --enable ol7_oracle_instantclient > /dev/null && \
    yum install -y -q \
    build-essential \
    bzip2 \
    curl \
    git \
    gfortran \
    libcurl-devel \
    libxml2-devel \
    oracle-instantclient${release}.${update}-basic \
    oracle-instantclient${release}.${update}-sqlplus \
    openssl \
    openssl-devel \
    patch \
    sudo \
    unzip \
    zip \
    g++ \
    wget \
    gcc \
    vim \
    yum groupinstall -y -q development tools \
    && yum clean all \
    && rm -rf /var/cache/yum/*


# setup user
RUN \
  mkdir -p /home/$DATASCIENCE_USER && \
  useradd -m -s /bin/bash -N -u $DATASCIENCE_UID $DATASCIENCE_USER && \
  chown -R $DATASCIENCE_USER /home/$DATASCIENCE_USER && \
  chown -R $DATASCIENCE_USER:users /usr/local/ && \
  touch /etc/sudoers.d/$DATASCIENCE_USER && echo "$DATASCIENCE_USER ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers.d/$DATASCIENCE_USER && \
  mkdir -p $DATASCIENCE_INSTALL_DIR && chown $DATASCIENCE_USER $DATASCIENCE_INSTALL_DIR && \
  mkdir -p $LOGS_DIRECTORY && chown -R $DATASCIENCE_USER:users $LOGS_DIRECTORY && \
  mkdir -p $LOGS_DIRECTORY/harness && chown -R $DATASCIENCE_USER:users $LOGS_DIRECTORY/harness
#   mkdir -p /home/$DATASCIENCE_USER/condapack


RUN mkdir -p /etc/datascience/build
RUN mkdir -p $DATASCIENCE_INSTALL_DIR/{pre-build-ds,post-build-ds,pre-run-ds,pre-run-user}

#conda
# set a default language for localization.  necessary for oci cli
ARG LANG=en_US.utf8
ENV LANG=$LANG
ENV SHELL=/bin/bash

# set /opt folder permissions for $DATASCIENCE_USER. Conda is going to live in this folder.
ARG MINICONDA_VER=4.8.3
RUN chown $DATASCIENCE_USER /opt

USER $DATASCIENCE_USER
WORKDIR /home/datascience
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> /home/datascience/miniconda.sh \
    && /bin/bash /home/datascience/miniconda.sh -f -b -p /opt/conda \
    && rm /home/datascience/miniconda.sh \
    && ls  /opt/**/* \
    && /opt/conda/bin/conda install python=3.7 anaconda \
    && /opt/conda/bin/conda clean -yaf

WORKDIR /
USER root
RUN printf "#!/bin/bash\nsource /opt/conda/bin/activate\n" > /etc/profile.d/enableconda.sh \
    && chmod +x /etc/profile.d/enableconda.sh

USER $DATASCIENCE_USER
ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /home/datascience


#COPY base-env.yaml /opt/base-env.yaml
#RUN conda env update -q -n root -f /opt/base-env.yaml && conda clean -yaf && rm -rf /home/datascience/.cache/pip


USER $DATASCIENCE_USER

####### WRAP UP ###############################
RUN python -c 'import sys; assert(sys.version_info[:2]) == (3, 7), "Python 3.7 is not detected"'
WORKDIR /

RUN conda list

USER root

############# Setup Conda environment tools ###########################
ARG RAND=1

RUN pip install conda_pack oci-cli

ARG RUN_WORKING_DIR="/home/datascience"
WORKDIR $RUN_WORKING_DIR

# clean tmp folder
RUN rm -rf /tmp/*

#COPY harness.py /etc/datascience/harness.py
RUN mkdir -p /etc/datascience/operators
# Temporarily removing operators related components
# COPY operators/run.py /etc/datascience/operators/run.py

USER datascience
