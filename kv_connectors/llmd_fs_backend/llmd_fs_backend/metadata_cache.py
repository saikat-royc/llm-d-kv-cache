# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash

from llmd_fs_backend import _logger as logger
from llmd_fs_backend.metrics import LLMD_FS_METADATA_CACHE_USAGE_BYTES


class MetadataCache:
    """
    A thread-safe, in-memory positive cache for KV block metadata.
    It uses an LRU eviction policy, but protects pinned entries (active I/O).
    """

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._lock = threading.RLock()
        # hash -> pin_count
        self._cache: OrderedDict[BlockHash, int] = OrderedDict()

        # Approximate size per entry: BlockHash (8 bytes) + pin_count (int, 8-28 bytes).
        # We'll use a conservative 64 bytes per entry for the OrderedDict overhead.
        self._bytes_per_entry = 64

    # ----------------------------------------------------------------------
    # Single Operations (Internal Helpers)
    # ----------------------------------------------------------------------

    def _insert(self, block_hash: BlockHash):
        """Internal insert logic without locking."""
        if block_hash in self._cache:
            self._cache.move_to_end(block_hash)
            return

        if len(self._cache) >= self.max_entries:
            # Try to evict the oldest unpinned entry
            evicted = False
            for k, pin_count in self._cache.items():
                if pin_count == 0:
                    self._cache.pop(k)
                    evicted = True
                    break

            if not evicted:
                logger.debug(
                    "Cache saturated with %d pinned entries. "
                    "Hit for %s will not be cached.",
                    len(self._cache),
                    block_hash.hex(),
                )
                return

        self._cache[block_hash] = 0

    def _pin(self, block_hash: BlockHash):
        """Internal pin logic without locking."""
        if block_hash not in self._cache:
            self._cache[block_hash] = 0
        self._cache[block_hash] += 1
        self._cache.move_to_end(block_hash)

    def _unpin(self, block_hash: BlockHash):
        """Internal unpin logic without locking."""
        if block_hash in self._cache:
            self._cache[block_hash] = max(0, self._cache[block_hash] - 1)
            logger.debug("Unpinned block %s, current pin_count=%d",
                         block_hash.hex(), self._cache[block_hash])

    def _remove(self, block_hash: BlockHash):
        """Internal remove logic without locking."""
        if block_hash in self._cache:
            self._cache.pop(block_hash)

    def _set_metrics(self, count: int):
        """Update the memory usage gauge (call outside of lock)."""
        LLMD_FS_METADATA_CACHE_USAGE_BYTES.set(count * self._bytes_per_entry)

    # ----------------------------------------------------------------------
    # Public Batch Operations
    # ----------------------------------------------------------------------

    def batch_contains(self, block_hashes: list[BlockHash]) -> list[bool]:
        """Check presence for a list of hashes in a single lock acquisition."""
        with self._lock:
            results = []
            for h in block_hashes:
                if h in self._cache:
                    self._cache.move_to_end(h)
                    results.append(True)
                else:
                    results.append(False)
            return results

    def batch_insert(self, block_hashes: Iterable[BlockHash]):
        """Insert multiple hashes in a single lock acquisition."""
        with self._lock:
            for h in block_hashes:
                self._insert(h)
            count = len(self._cache)
        self._set_metrics(count)

    def batch_pin(self, block_hashes: Iterable[BlockHash]):
        """Pin multiple hashes in a single lock acquisition."""
        with self._lock:
            for h in block_hashes:
                self._pin(h)
            count = len(self._cache)
        self._set_metrics(count)

    def batch_unpin(self, block_hashes: Iterable[BlockHash]):
        """Unpin multiple hashes in a single lock acquisition."""
        with self._lock:
            for h in block_hashes:
                self._unpin(h)

    def batch_remove(self, block_hashes: Iterable[BlockHash]):
        """Remove multiple hashes in a single lock acquisition."""
        with self._lock:
            for h in block_hashes:
                self._remove(h)
            count = len(self._cache)
        self._set_metrics(count)

    # ----------------------------------------------------------------------
    # Compatibility Single Accessors
    # ----------------------------------------------------------------------

    def contains(self, block_hash: BlockHash) -> bool:
        with self._lock:
            if block_hash in self._cache:
                self._cache.move_to_end(block_hash)
                return True
            return False

    def insert(self, block_hash: BlockHash):
        with self._lock:
            self._insert(block_hash)
            count = len(self._cache)
        self._set_metrics(count)

    def remove(self, block_hash: BlockHash):
        with self._lock:
            self._remove(block_hash)
            count = len(self._cache)
        self._set_metrics(count)

    def pin(self, block_hash: BlockHash):
        with self._lock:
            self._pin(block_hash)
            count = len(self._cache)
        self._set_metrics(count)

    def unpin(self, block_hash: BlockHash):
        with self._lock:
            self._unpin(block_hash)
