"""Verify the locally cached BGE-M3 embedding model and CUDA availability."""
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
CACHE_DIR = Path(__file__).resolve().parents[1] / "storage" / "models"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        cache_folder=str(CACHE_DIR),
        local_files_only=True,
    )
    vectors = model.encode(
        [
            "Grid Computing coordinates distributed computing resources to solve tasks.",
            "网格计算通过多个计算节点协同处理任务。",
            "我的笔记：云计算向用户隐藏底层基础设施细节。",
        ],
        normalize_embeddings=True,
        batch_size=2,
        show_progress_bar=False,
    )
    print(f"model={MODEL_NAME}")
    print(f"device={device}")
    print(f"vector_shape={vectors.shape}")
    print(f"cache_dir={CACHE_DIR}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"max_gpu_memory_mb={torch.cuda.max_memory_allocated(0) / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
