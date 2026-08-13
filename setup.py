import os
import sys
import argparse

SD_MIRROR = "Manojb/stable-diffusion-2-1-base"
SD_LOCAL_DIR = os.path.join("weights", "stable-diffusion-2-1-base")


def download_sd_weights():
    """Download Stable Diffusion 2.1-base weights in diffusers format.

    NOTE: The official repository (stabilityai/stable-diffusion-2-1-base) was
    made private by Stability AI in late 2025. We instead use a community
    mirror (Manojb/stable-diffusion-2-1-base) that hosts the same weights.
    """
    if os.path.exists(os.path.join(SD_LOCAL_DIR, "model_index.json")):
        print(f"StableDiffusion v2.1 weights already exist: {SD_LOCAL_DIR}")
        return

    print("NOTE: StableDiffusion v2.1 weights not found. Downloading...")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=SD_MIRROR,
            local_dir=SD_LOCAL_DIR,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "tokenizer/*",
                "unet/*",
                "vae/*",
            ],
            ignore_patterns=["*.ckpt", "*.bin"],
        )
        print(f"Download completed: {SD_LOCAL_DIR}")
    except Exception as e:
        print(f"Error: download failed ({e})")
        print(f"Please manually download from https://huggingface.co/{SD_MIRROR}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up directories and pretrained weights for NOLA-IR.")
    parser.add_argument(
        "--download_sd_weights",
        action="store_true",
        help="Download Stable Diffusion 2.1-base weights (diffusers format)",
    )
    args = parser.parse_args()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("datasets/source", exist_ok=True)

    os.makedirs("experiments/classification/cub200", exist_ok=True)
    os.makedirs("experiments/segmentation/voc2012", exist_ok=True)
    os.makedirs("experiments/detection/voc2012", exist_ok=True)

    if args.download_sd_weights:
        download_sd_weights()

    print("Setup Done")
