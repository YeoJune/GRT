from grt.config import validate_model


def create_model(cfg):
    validate_model(cfg.model)
    if cfg.model.name == "rmt":
        from .rmt import make_rmt

        return make_rmt(cfg)
    from .grt import GRTModel

    return GRTModel(cfg.model)
