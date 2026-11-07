import sys, os
import json
CONFIG = {"debug": True}
cache = {}
def load(path):
    data = json.loads(open(path).read())
    cache[path] = data
    return data
def save(path, obj):
    text = json.dumps(obj)
    open(path, "w").write(text)
class Store:
    """Простое хранилище.

    С пустыми строками
    внутри docstring.
    """
    def __init__(self, root):
        self.root = root
        self.items = {}
    def add(self, key, value):
        self.items[key] = value
        if len(self.items) > 100:
            self._evict()
        return True
    def _evict(self):
        keys = sorted(self.items)
        victim = keys[0]
        del self.items[victim]
