# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

FROM ghcr.io/oracle/oraclelinux7-instantclient:19 AS base

RUN rm -rf /var/cache/yum/* && yum clean all && yum install -y gcc mesa-libGL vim && rm -rf /var/cache/yum/*
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:$PATH"

ENV HOME /home/datascience
RUN mkdir -p /etc/datascience
WORKDIR /etc/datascience

COPY operators/environment.yaml operators/environment.yaml
RUN conda env create -f operators/environment.yaml --name op_env && conda clean -afy
ENV PATH="/miniconda/envs/op_env/bin:$PATH"

RUN /bin/bash -c "source activate op_env"
COPY operators/run.py operators/run.py
CMD bash

FROM base
COPY docker/merge_dependencies.py merge_dependencies.py
COPY operators/ operators/

RUN pip install pyyaml click && python merge_dependencies.py environment.yaml
RUN conda env update -f environment.yaml --name op_env && conda clean -afy
RUN source activate op_env
