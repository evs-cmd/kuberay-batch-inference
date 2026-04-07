"""
Batch inference job — submitted via Ray Job SDK.

Usage:
    ray job submit --address http://localhost:8265 \
        --working-dir . \
        -- python jobs/batch_job.py

    # With custom prompts:
    ray job submit --address http://localhost:8265 \
        --working-dir . \
        -- python jobs/batch_job.py \
        --prompts "What is 2+2?" "Explain gravity" "Hello world"

    # Check status:
    ray job status <job_id>
    ray job logs <job_id>
"""

import ray
import json
import time
import argparse
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="Batch inference via Ray")
    parser.add_argument(
        "--prompts", nargs="+",
        default=["What is 2+2?", "Hello world", "Explain gravity in one sentence"],
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    ray.init()
    print(f"Cluster resources: {ray.cluster_resources()}")

    from app.inference import BatchOrchestrator

    orchestrator = BatchOrchestrator(model_name=args.model)

    start = time.time()
    results = orchestrator.run_batch(args.prompts, args.max_tokens)
    elapsed = time.time() - start

    output = {
        "status": "completed",
        "model": args.model,
        "total_prompts": len(args.prompts),
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(
            sum(r["tokens_generated"] for r in results) / elapsed, 1
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    print(json.dumps(output, indent=2))

    orchestrator.shutdown()


if __name__ == "__main__":
    main()
