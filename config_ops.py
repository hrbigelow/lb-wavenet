# Taken from 3b1b/manim

def digest_config(obj, kwargs, caller_locals={}):
    """
    Sets init args and CONFIG values as local variables

    The purpose of this function is to ensure that all
    configuration of any object is inheritable, able to
    be easily passed into instantiation, and is attached
    as an attribute of the object.
    """

    # Assemble list of CONFIGs from all super classes
    classes_in_hierarchy = [obj.__class__]
    static_configs = []
    while len(classes_in_hierarchy) > 0:
        Class = classes_in_hierarchy.pop()
        classes_in_hierarchy += Class.__bases__
        if hasattr(Class, "CONFIG"):
            static_configs.append(Class.CONFIG)

    # Order matters a lot here, first dicts have higher priority
    caller_locals = filtered_locals(caller_locals)
    all_dicts = [kwargs, caller_locals, obj.__dict__]
    all_dicts += static_configs
    obj.__dict__ = merge_config(all_dicts)


def merge_config(all_dicts):
    all_config = reduce(op.add, [list(d.items()) for d in all_dicts])
    config = dict()
    for c in all_config:
        key, value = c
        if key not in config:
            config[key] = value
        else:
            # When two dictionaries have the same key, they are merged.
            if isinstance(value, dict) and isinstance(config[key], dict):
                config[key] = merge_config([config[key], value])
    return config

