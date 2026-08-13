import os
from argparse import ArgumentParser
from importlib import import_module
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str) -> Any:
    cfg = OmegaConf.load(path)
    config_dir = os.path.dirname(os.path.abspath(path))
    for key in ("dataset",):
        if isinstance(cfg.get(key), str):
            cfg[key] = OmegaConf.load(cfg[key])
    return cfg


def build_pipeline(cfg: Any, args: Any) -> Any:
    parts = cfg.pipeline.split(".")
    module_path, attr = ".".join(parts[:-1]), parts[-1]
    pipeline = getattr(import_module(module_path), attr)
    return pipeline(cfg, args)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["detection", "ocr"], default="detection")
    parser.add_argument("--scale", type=float, default=None,
                        help="manual resize factor (detection only). "
                             "if unset, the long side is resized to 512. "
                             "ignored for OCR, which always resizes to a fixed width of 512"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    if args.task == "detection":
        args.config = "configs/detection/nola-ir-demo.yaml"
    elif args.task == "ocr":
        args.config = "configs/optical_character_recognition/nola-ir-demo.yaml"
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg, args)
    pipeline.evaluate()


if __name__ == "__main__":
    main()
