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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.get("train"):
        args.eval_only = True
    pipeline = build_pipeline(cfg, args)

    if args.eval_only:
        pipeline.evaluate()
    else:
        pipeline.train()


if __name__ == "__main__":
    main()
