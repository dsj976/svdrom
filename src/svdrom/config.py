_global_config = {
    "hankel_coord_name": "lag",
    "stack_coord_name": "samples",
    "precision": "single",
    "logger": True,
}


def get(key=None):
    """Get the current configuration.
    If key is None, returns the whole config dict.
    Otherwise, returns the value for the given key.
    """
    if key is None:
        return _global_config.copy()
    return _global_config.get(key)


def set(**kwargs):
    """Update configuration values for the current session.

    Examples
    --------
    >>> import svdrom.config

    Use double instead of single precision:
    >>> svdrom.config.set(precision="double")

    When stacking dimensions together, call the
    resulting dimension "space":
    >>> svdrom.config.set(stack_coord_name="space")
    """
    for k, v in kwargs.items():
        if k in _global_config:
            _global_config[k] = v
        else:
            msg = f"Unknown config key: {k}"
            raise KeyError(msg)
