"""
Qwen model server — download, load, generate.

Used by Ray workers (inference.py) and as standalone verification:
    python -m app.model_server
    python -m app.model_server --download-only
"""

import time
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def download_model(model_name: str = DEFAULT_MODEL) -> str:
    """Download model weights from HuggingFace. Returns local path."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading {model_name}...")
    path = snapshot_download(model_name)
    logger.info(f"Cached at: {path}")
    return path


class ModelServer:
    """
    Wraps vLLM with a simple generate() interface.
    One instance per Ray worker — model stays in VRAM across calls.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from vllm import LLM

        self.model_name = model_name
        download_model(model_name)

        start = time.time()
        logger.info(f"Loading {model_name} into VRAM...")
        self.engine = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=2048,
        )
        self.load_time = time.time() - start
        logger.info(f"Model loaded in {self.load_time:.1f}s")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.engine.generate(prompts, params)

        return [
            {
                "prompt": out.prompt,
                "generated_text": out.outputs[0].text,
                "tokens_generated": len(out.outputs[0].token_ids),
                "finish_reason": out.outputs[0].finish_reason or "unknown",
            }
            for out in outputs
        ]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    if args.download_only:
        download_model(args.model)
    else:
        server = ModelServer(args.model)
        start = time.time()
        results = server.generate(["What is 2+2?"], max_tokens=20)
        elapsed = time.time() - start
        r = results[0]
        print(f"Output: {r['generated_text'].strip()}")
        print(f"Tokens: {r['tokens_generated']} in {elapsed:.2f}s")
        print(f"Verification: PASSED")
