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

import os
import time
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.metadata_cache import MetadataCache
from llmd_fs_backend.metrics import (
    LLMD_FS_LOOKUP_DURATION_SECONDS,
    LLMD_FS_LOOKUP_HIT_TOTAL,
    LLMD_FS_LOOKUP_TOTAL_BLOCKS,
)

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    def __init__(self, file_mapper: FileMapper, metadata_cache_size: int = 0) -> None:
        self.file_mapper: FileMapper = file_mapper
        self.cache: MetadataCache | None = (
            MetadataCache(metadata_cache_size) if metadata_cache_size > 0 else None
        )

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """
        Return how many consecutive blocks from the start are already offloaded.
        """
        start_time = time.monotonic()

        # Materialize the list to get a stable count for metrics
        hashes_list = list(block_hashes)
        total_requested = len(hashes_list)
        LLMD_FS_LOOKUP_TOTAL_BLOCKS.inc(total_requested)

        # 1. Batch check in-memory cache
        if self.cache:
            cache_hits = self.cache.batch_contains(hashes_list)
        else:
            cache_hits = [False] * total_requested

        hit_count = 0
        memory_hits = 0
        fs_hits = 0
        new_hits_to_cache = []

        # 2. Iterate and fallback to FS only when needed
        for i, is_cache_hit in enumerate(cache_hits):
            if is_cache_hit:
                hit_count += 1
                memory_hits += 1
                continue

            # Fallback to filesystem for consecutive blocks
            block_hash = hashes_list[i]
            file_path = self.file_mapper.get_file_name(block_hash)
            if not os.path.exists(file_path):
                # First missing block found - stop lookup
                break

            hit_count += 1
            fs_hits += 1
            new_hits_to_cache.append(block_hash)

        # 3. Batch update cache with new hits found from FS
        if self.cache and new_hits_to_cache:
            self.cache.batch_insert(new_hits_to_cache)

        duration = time.monotonic() - start_time

        # Update Prometheus metrics
        LLMD_FS_LOOKUP_DURATION_SECONDS.labels(
            num_blocks=str(total_requested)).observe(duration)
        LLMD_FS_LOOKUP_HIT_TOTAL.inc(hit_count)

        # Calculate overlap percentage (handling empty requests)
        overlap_pct = (hit_count / total_requested * 100) if total_requested > 0 else 0.0

        # Emit metrics to the log
        logger.info(
            "Lookup finished: duration=%.6f [s] overlap=%.2f%% "
            "hits=%d/%d blocks (memory=%d, filesystem=%d)",
            duration,
            overlap_pct,
            hit_count,
            total_requested,
            memory_hits,
            fs_hits,
        )

        return hit_count

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        if self.cache:
            self.cache.batch_pin(block_hashes)

        return SharedStorageLoadStoreSpec(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Update access times if desired.
        Shared storage version does nothing here because updates are handled
        by the file thread for performance reasons.
        """
        pass

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """Stateless load - no post-load action needed, but unpin metadata."""
        if self.cache:
            self.cache.batch_unpin(block_hashes)

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Prepare storing new blocks.
        Shared storage always accepts new blocks. Eviction is not needed.
        If a file already exists, the file thread handles it.
        """
        block_hashes_to_store = list(block_hashes)

        if self.cache:
            self.cache.batch_pin(block_hashes_to_store)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(block_hashes_to_store)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=[],  # no eviction needed
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """
        For shared storage, storing is stateless - no action needed, but unpin.
        """
        if self.cache:
            if success:
                # Ensure it is in the positive cache after successful store
                self.cache.batch_insert(block_hashes)
            self.cache.batch_unpin(block_hashes)
