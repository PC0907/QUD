from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config; if it declares `inherit: <file>`, the parent
    (resolved relative to the config) is loaded first and shallowly
    overridden."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(path.parent / parent)
        base.update(cfg)
        return base
    return cfg
