FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ARG PYTHON_VERSION=3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

ENV CHAINER_BUILD_CHAINERX=1
ENV CHAINERX_BUILD_CUDA=1
COPY requirements.txt .
RUN pip install -r requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/

RUN apt-get update && apt-get -y install gosu

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]