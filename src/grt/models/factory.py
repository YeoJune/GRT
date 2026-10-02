from grt.config import validate_model

def create_model(cfg):
    validate_model(cfg)
    if cfg.name == "grt":
        from .grt import GRTModel
        return GRTModel(cfg)
    if cfg.name == "rmt":
        from .rmt import RMTModel
        return RMTModel(cfg)
    raise ValueError(f"Unknown model: {cfg.name}")
