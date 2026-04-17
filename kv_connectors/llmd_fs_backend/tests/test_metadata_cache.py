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
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.metadata_cache import MetadataCache
from llmd_fs_backend.metrics import LLMD_FS_METADATA_CACHE_USAGE_BYTES
from llmd_fs_backend.spec import SharedStorageOffloadingSpec


def test_metadata_cache_basic():
    """Test basic insert and contains logic."""
    cache = MetadataCache(max_entries=10)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")

    assert not cache.contains(h1)
    cache.insert(h1)
    assert cache.contains(h1)
    assert not cache.contains(h2)


def test_metadata_cache_lru_eviction():
    """Test LRU eviction policy."""
    cache = MetadataCache(max_entries=2)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache.insert(h1)
    cache.insert(h2)
    assert cache.contains(h1)
    assert cache.contains(h2)

    # Insert h3, should evict h1 (since h2 was most recently accessed via contains)
    cache.insert(h3)
    assert not cache.contains(h1)
    assert cache.contains(h2)
    assert cache.contains(h3)


def test_metadata_cache_pinning():
    """Test that pinned entries are not evicted."""
    cache = MetadataCache(max_entries=2)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache.insert(h1)
    cache.insert(h2)
    cache.pin(h1)  # Pin the oldest

    # Insert h3, should evict h2 because h1 is pinned
    cache.insert(h3)
    assert cache.contains(h1)
    assert not cache.contains(h2)
    assert cache.contains(h3)

    # Unpin h1 and insert h2, now h1 can be evicted
    cache.unpin(h1)
    cache.insert(h2)
    assert not cache.contains(h1)
    assert cache.contains(h3)
    assert cache.contains(h2)


def test_metadata_cache_saturation():
    """Test that hits are not cached when the cache is full of pinned entries."""
    cache = MetadataCache(max_entries=2)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    cache.pin(h1)
    cache.pin(h2)

    with patch("llmd_fs_backend.metadata_cache.logger.debug") as mock_debug:
        cache.insert(h3)
        mock_debug.assert_called_once()
        assert "Cache saturated" in mock_debug.call_args[0][0]

    assert cache.contains(h1)
    assert cache.contains(h2)
    assert not cache.contains(h3)


def test_metadata_cache_usage_metrics():
    """Test that the usage gauge is updated."""
    # Reset gauge for test
    LLMD_FS_METADATA_CACHE_USAGE_BYTES.set(0)

    cache = MetadataCache(max_entries=10)
    h1 = BlockHash(b"hash1111")

    cache.insert(h1)
    assert LLMD_FS_METADATA_CACHE_USAGE_BYTES._value.get() == 64

    cache.remove(h1)
    assert LLMD_FS_METADATA_CACHE_USAGE_BYTES._value.get() == 0


def test_manager_lookup_fallthrough():
    """Test manager lookup with cache hits and FS fallthrough."""
    file_mapper = MagicMock(spec=FileMapper)
    file_mapper.get_file_name.return_value = "/tmp/fake_path"
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_size=10)

    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")

    # Mock os.path.exists
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        # First lookup: H1 should hit FS and be cached
        assert manager.lookup([h1]) == 1
        assert mock_exists.call_count == 1
        assert manager.cache.contains(h1)

        # Second lookup: H1 should hit cache, FS should NOT be called
        assert manager.lookup([h1]) == 1
        assert mock_exists.call_count == 1

        # Lookup with H2 (not in cache)
        assert manager.lookup([h1, h2]) == 2
        assert mock_exists.call_count == 2
        assert manager.cache.contains(h2)


def test_complete_store_pins():
    """Test that complete_store unpins and inserts into cache."""
    file_mapper = MagicMock(spec=FileMapper)
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_size=10)
    h1 = BlockHash(b"hash1111")

    # Store sequence
    manager.prepare_store([h1])
    assert manager.cache._cache[h1] == 1

    manager.complete_store([h1], success=True)
    assert manager.cache._cache[h1] == 0
    assert manager.cache.contains(h1)


def test_spec_env_var_config():
    """Test that environment variable overrides cache size."""
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.model = "test-model"
    vllm_config.cache_config.cache_dtype = "float16"
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.world_size = 1
    vllm_config.parallel_config.rank = 0

    kv_config = MagicMock(spec=KVCacheConfig)
    # Patch get_kv_cache_group_block_size to return [16]
    with patch("vllm.v1.kv_offload.spec.OffloadingSpec.gpu_block_size", [16]):
        # Test Default (disabled)
        with patch.dict(os.environ, {}, clear=True):
            spec = SharedStorageOffloadingSpec(vllm_config, kv_config)
            assert spec.metadata_cache_size == 0

        # Test Env Var Override
        with patch.dict(
            os.environ, {"VLLM_LLMD_FS_METADATA_CACHE_SIZE": "500000"}
        ):
            spec = SharedStorageOffloadingSpec(vllm_config, kv_config)
            assert spec.metadata_cache_size == 500000

        # Verify Manager initialization
        manager = spec.get_manager()
        assert manager.cache is not None
        assert manager.cache.max_entries == 500000


def test_metadata_cache_batch_ops():
    """Test batch contains, insert, pin, and unpin."""
    cache = MetadataCache(max_entries=10)
    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    # Batch Insert
    cache.batch_insert([h1, h2])
    results = cache.batch_contains([h1, h2, h3])
    assert results == [True, True, False]

    # Batch Pin
    cache.batch_pin([h1, h2, h3])
    # h3 was not in cache, batch_pin should have added it
    assert cache.contains(h3)
    assert cache._cache[h1] == 1
    assert cache._cache[h2] == 1
    assert cache._cache[h3] == 1

    # Batch Unpin
    cache.batch_unpin([h1, h2])
    assert cache._cache[h1] == 0
    assert cache._cache[h2] == 0
    assert cache._cache[h3] == 1


def test_manager_lookup_batch_optimized():
    """Test that manager lookup uses batch operations and hits FS only for misses."""
    file_mapper = MagicMock(spec=FileMapper)
    file_mapper.get_file_name.side_effect = lambda h: f"/tmp/{h.hex()}"
    manager = SharedStorageOffloadingManager(file_mapper, metadata_cache_size=10)

    h1 = BlockHash(b"hash1111")
    h2 = BlockHash(b"hash2222")
    h3 = BlockHash(b"hash3333")

    manager.cache.insert(h1)

    with patch("os.path.exists") as mock_exists:
        # H1 is in cache, H2 is in FS, H3 is NOT in FS
        def exists_side_effect(path):
            return "hash2222" in path

        mock_exists.side_effect = exists_side_effect

        # Lookup [H1, H2, H3]
        # Should:
        # 1. Batch check cache: [H1: True, H2: False, H3: False]
        # 2. Skip H1 (cache hit)
        # 3. Check H2 via FS (exists)
        # 4. Check H3 via FS (not exists) -> stop
        hit_count = manager.lookup([h1, h2, h3])

        assert hit_count == 2
        assert mock_exists.call_count == 2  # H2 and H3
        assert manager.cache.contains(h2)
        assert not manager.cache.contains(h3)
