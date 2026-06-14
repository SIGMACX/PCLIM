"""CMPCN implementation for EP_FCHD fetal cardiac ultrasound experiments."""

from .config import TrainConfig
from .text_descriptions import EP_FCHD_CLASS_DESCRIPTIONS, EP_FCHD_CLASS_NAMES

__all__ = [
    "TrainConfig",
    "CMPCNModel",
    "build_cmpcn",
    "EP_FCHD_CLASS_DESCRIPTIONS",
    "EP_FCHD_CLASS_NAMES",
]


def __getattr__(name):
    if name in {"CMPCNModel", "build_cmpcn"}:
        from .model import CMPCNModel, build_cmpcn

        return {"CMPCNModel": CMPCNModel, "build_cmpcn": build_cmpcn}[name]
    raise AttributeError(name)
