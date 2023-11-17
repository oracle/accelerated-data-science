# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

FROM ghcr.io/oracle/oraclelinux8-instantclient:21 as base

RUN \
    rm -rf /var/cache/yum/* && \
    yum install -y gcc make patch vim iproute net-tools git && \
    yum clean all && \
    rm -rf /var/cache/yum/*

########################### CONDA INSTALLATION ########################################
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:$PATH"

USER root

ARG CONDA_ENV_FILE=environment.yaml
ARG CONDA_ENV_NAME=operator
ARG OPERATOR_DIR=/etc/operator
ARG RND

COPY ./${CONDA_ENV_FILE} /opt/env.yaml
RUN conda install conda-forge::mamba && mamba env create -f /opt/env.yaml --name ${CONDA_ENV_NAME} && conda clean -afy
ENV PATH="/miniconda/envs/${CONDA_ENV_NAME}}/bin:$PATH"

RUN conda init bash && source ~/.bashrc && conda activate ${CONDA_ENV_NAME}

########################### SETUP WORKDIR ########################################
RUN mkdir ${OPERATOR_DIR}

ENV OPERATOR_DIR=${OPERATOR_DIR}
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}

WORKDIR ${OPERATOR_DIR}

RUN echo "conda activate $CONDA_ENV_NAME">>/root/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENTRYPOINT [ "bash", "--login" , "-c"]
