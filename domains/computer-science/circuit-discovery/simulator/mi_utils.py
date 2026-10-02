import types


def patch_transformer_lens_rope_theta():
    try:
        import transformer_lens.loading_from_pretrained as loading
    except ImportError:
        return
    if getattr(loading, "rope_theta_patched", False):
        return
    original = loading.AutoConfig.from_pretrained

    def from_pretrained(*args, **kwargs):
        cfg = original(*args, **kwargs)
        if not hasattr(cfg, "rope_theta"):
            theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
            if theta is not None:
                cfg.rope_theta = theta
        return cfg

    loading.AutoConfig = types.SimpleNamespace(from_pretrained=from_pretrained)
    loading.rope_theta_patched = True
