"""Verify that the locally downloaded Qwen3-ASR model can run on this PC."""

from __future__ import annotations

from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "storage" / "models" / "Qwen3-ASR-0.6B"
SAMPLE_AUDIO = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
)


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model is missing: {MODEL_DIR}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"model={MODEL_DIR}")
    print(f"device={device}")

    model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=128,
    )
    result = model.transcribe(audio=SAMPLE_AUDIO, language="English")[0]
    print(f"language={result.language}")
    print(f"text={result.text}")
    if device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(
            f"max_gpu_memory_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}"
        )


if __name__ == "__main__":
    main()
