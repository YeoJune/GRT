from grt.config import validate_model

def create_model(cfg):
    validate_model(cfg)
    if cfg.name == "grt":
        from .grt import GRTModel
        return GRTModel(cfg)
    if cfg.name == "rmt":
        if cfg.rmt_backbone == "author_neox":
            raise ValueError("Author RMT uses grt.reference.runner and its native labels/loss interface")
        if cfg.rmt_backbone == "gpt_neox":
            from .rmt_neox import NeoXRMTModel
            return NeoXRMTModel(cfg)
        if cfg.rmt_backbone == "relative_postln":
            from .rmt_relative import RelativeRMTModel
            return RelativeRMTModel(cfg)
        from .rmt import RMTModel
        return RMTModel(cfg)
    raise ValueError(f"Unknown model: {cfg.name}")
