from grt.config import validate_model
from .neox import RecurrentNeoX


def create_model(cfg):
    validate_model(cfg.model)
    return RecurrentNeoX(cfg)
