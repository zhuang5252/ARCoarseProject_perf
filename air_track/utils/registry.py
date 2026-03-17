from collections import defaultdict


class Registry:
    def __init__(self):
        self._modules = {}

    def register(self, name=None):
        def decorator(fn):
            key = name or fn.__name__
            if key in self._modules:
                raise KeyError(f"Module '{key}' is already registered.")
            self._modules[key] = fn
            return fn

        return decorator

    def get(self, name):
        if name not in self._modules:
            raise KeyError(f"Module '{name}' not found. Available: {list(self._modules)}")
        return self._modules[name]

    def list(self):
        return list(self._modules.keys())


# 定义统一注册器（项目全局可用）
BACKBONES = Registry()
HEADS = Registry()
DATASETS = Registry()
