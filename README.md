# KubeRay Batch Inference — Qwen2.5-0.5B

Distributed offline batch inference using Ray + vLLM + FastAPI.

## Quick start

```bash
git clone https://github.com/evs-cmd/kuberay-batch-inference.git
cd kuberay-batch-inference
make deploy
```

This installs dependencies, downloads the model, starts Ray + FastAPI, and runs a test. One command.

## Submit a batch

```bash
curl -X POST http://localhost:8000/v1/batches \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-kuberay-batch-2024" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": [{"prompt": "What is 2+2?"}, {"prompt": "Hello world"}],
    "max_tokens": 50
  }'
```

Poll for results:

```bash
curl http://localhost:8000/v1/batches/{JOB_ID} \
  -H "X-API-Key: sk-kuberay-batch-2024"
```

## Submit via Ray Job SDK

```bash
make job

# Or with custom prompts:
ray job submit --address http://localhost:8265 --working-dir . \
  -- python jobs/batch_job.py --prompts "Explain gravity" "What is DNA?"
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/batches` | Submit batch job |
| `GET` | `/v1/batches/{job_id}` | Get status + results |
| `GET` | `/v1/batches` | List all jobs |
| `GET` | `/health` | Health check (no auth) |
| `GET` | `/metrics` | Prometheus metrics (no auth) |

Auth: `X-API-Key: sk-kuberay-batch-2024` on all routes except `/health` and `/metrics`.

Model aliases: `Qwen/Qwen2.5-14B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` both resolve to the deployed model.

## Architecture

```
Client → FastAPI (auth, job mgmt) → Ray Scheduler → GPU Workers (vLLM) → Results
```

- **FastAPI** receives batch requests, returns job IDs, stores results to `/tmp/batch_results/`
- **Ray** distributes prompts as micro-batches (size 2) round-robin across GPU workers
- **vLLM** runs inside each Ray actor with PagedAttention + continuous batching
- **Micro-batching** mitigates the straggler problem (batch time = slowest worker)

### Task distribution

Prompts are chunked into micro-batches of 2 and distributed round-robin:

```
Input: [P1, P2, P3, P4, P5, P6, P7, P8]
         ↓ chunk into micro-batches of 2
       [P1,P2]  [P3,P4]  [P5,P6]  [P7,P8]
         ↓        ↓        ↓        ↓
      Worker 1  Worker 2  Worker 1  Worker 2  (round-robin)
```

Small chunks give Ray scheduling granularity to rebalance around variable-length prompts.

### KubeRay deployment (production)

Config files in `k8s/` for deploying on Kubernetes via KubeRay:

```bash
# 1. Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator -f k8s/kuberay-values.yaml

# 2. Build and push image
docker build -t vibs94/kuberay-batch-inference:latest .
docker push vibs94/kuberay-batch-inference:latest
or
minikube image build -t kuberay-batch-inference:latest .

# 3. Deploy cluster + services
kubectl apply -f k8s/raycluster.yaml
kubectl apply -f k8s/service.yaml

# 4. Submit a job (K8s-native)
kubectl apply -f k8s/rayjob.yaml
kubectl logs -l job-name=batch-inference-job

# 5. Or submit via Ray Job SDK
kubectl port-forward svc/batch-inference-api 8265:8265
ray job submit --address http://localhost:8265 --working-dir . -- python jobs/batch_job.py
```

**Strengths:** Native CRDs, worker autoscaling (1-8 GPUs), auto pod restart, Ray Dashboard as K8s service, health/liveness probes, shared model cache via hostPath.

**Limitations:** Head node SPOF (mitigate with GCS fault tolerance + Redis), no GPU sharing (MIG), cold start on scale-up, no built-in job priority queue.

## Project structure

```
app/
├── main.py              # FastAPI gateway
├── models.py            # Pydantic schemas
├── model_server.py      # Model download + load + generate
└── inference.py         # Ray workers + batch orchestrator
jobs/
└── batch_job.py         # Ray Job SDK submission
k8s/                     # KubeRay production config
├── kuberay-values.yaml  # Helm values for KubeRay operator
├── raycluster.yaml      # RayCluster CR (head + GPU workers)
├── rayjob.yaml          # RayJob CR (K8s-native job submission)
└── service.yaml         # Services (API, dashboard, head)
requirements/
└── gpu.txt
scripts/
├── deploy.sh            # Install + start everything
Makefile
Dockerfile               # For container-based deployment
```

## Make targets

```
  deploy         Install deps, start Ray + FastAPI, test
  stop           Stop Ray + FastAPI
  test           Run integration tests
  verify         Download + verify model
  job            Submit batch via Ray Job SDK
```
