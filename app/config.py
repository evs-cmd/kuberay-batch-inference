import os
from pathlib import Path

API_KEY = os.getenv("API_KEY", "sk-kuberay-batch-2024")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/tmp/batch_results"))
