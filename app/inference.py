"""
Ray-distributed inference.

Each worker is a persistent Ray actor holding a ModelServer instance.
Prompts are chunked into micro-batches and distributed round-robin
for straggler mitigation.
"""

import logging
from typing import Optional

import ray
from app.model_server import ModelServer

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InferenceWorker:
    """Persistent Ray actor — model stays in VRAM across calls."""

    def __init__(self, model_name: str):
        self.server = ModelServer(model_name=model_name)

    def generate(self, prompts: list[str], max_tokens: int = 50) -> list[dict]:
        return self.server.generate(prompts, max_tokens)

    def health_check(self) -> dict:
        return {"model": self.server.model_name, "status": "healthy"}


class BatchOrchestrator:
    """
    Manages a pool of workers and distributes batches.

    Distribution strategy:
    - Chunk prompts into micro-batches of 2
    - Assign round-robin across workers
    - Small chunks let Ray rebalance around stragglers
    - vLLM handles continuous batching within each worker
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
            num_workers: Optional[int] = None,
            micro_batch_size: int = 2,
    ):
        self.model_name = model_name
        self.micro_batch_size = micro_batch_size

        if num_workers is None:
            resources = ray.cluster_resources()
            num_workers = max(1, int(resources.get("GPU", 1)))

        logger.info(f"Starting {num_workers} worker(s) for {model_name}")
        self.workers = [InferenceWorker.remote(model_name) for _ in range(num_workers)]

        # Wait for models to load
        ray.get([w.health_check.remote() for w in self.workers])
        logger.info(f"All {num_workers} worker(s) ready.")

    def run_batch(self, prompts: list[str], max_tokens: int = 50) -> list[dict]:
        """Chunk into micro-batches, distribute round-robin, collect results."""
        chunks = [
            prompts[i: i + self.micro_batch_size]
            for i in range(0, len(prompts), self.micro_batch_size)
        ]

        futures = [
            self.workers[idx % len(self.workers)].generate.remote(chunk, max_tokens)
            for idx, chunk in enumerate(chunks)
        ]

        all_results = []
        for result_batch in ray.get(futures):
            all_results.extend(result_batch)
        return all_results

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
