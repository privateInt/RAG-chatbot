# 빌드 스테이지
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /workspace

COPY . /workspace

EXPOSE 18000

# Python 3.10, pip, 그리고 Git 설치
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y \
        python3.10 \
        python3.10-distutils \
        python3.10-dev \
        curl \
        git \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 파이썬으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3

# get-pip.py 다운로드 및 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py

# 필요한 라이브러리 설치
RUN python3.10 -m pip install -r requirements.txt

# 버전 업
RUN python3.10 -m pip install -U torch transformers peft

CMD ["/bin/bash"]