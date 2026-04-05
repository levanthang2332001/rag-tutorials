"""TTL-bounded LRU caches shared by tools."""

import threading
import time
from collections import OrderedDict


class _BoundedTtlCache:
    def __init__(self, max_items: int, ttl_s: float):
        self._max_items = max_items
        self._ttl_s = ttl_s
        self._items: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created_at, value = item
            if now - created_at > self._ttl_s:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: str) -> None:
        now = time.time()
        with self._lock:
            self._items[key] = (now, value)
            self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)


_WIKIPEDIA_CACHE = _BoundedTtlCache(max_items=256, ttl_s=60 * 30)
