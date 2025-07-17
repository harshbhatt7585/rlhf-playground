FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


COPY . /app


ENTRYPOINT ["python", "train.py"]
CMD []