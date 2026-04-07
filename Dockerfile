# Dockerfile — KubeRay Batch Inference
# Base image already has Ray 2.35.0 + CUDA 12.1 + Python 3.11
#
# Build inside minikube:
#   minikube image build -t kuberay-batch:latest .

FROM rayproject/ray:2.35.0-py311-cu121

WORKDIR /app

COPY requirements/gpu.txt requirements/gpu.txt
RUN pip install --no-cache-dir -r requirements/gpu.txt

COPY app/ app/
COPY jobs/ jobs/

RUN mkdir -p /tmp/batch_results

EXPOSE 8000 8265 6379