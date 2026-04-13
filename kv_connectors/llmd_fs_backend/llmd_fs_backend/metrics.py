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

from prometheus_client import Counter, Histogram

# Prometheus metrics for llmd_fs_backend
LLMD_FS_LOOKUP_DURATION_SECONDS = Histogram(
    "vllm_llmd_fs_lookup_duration_seconds",
    "Time spent in the metadata lookup function of the fs backend manager.",
)

LLMD_FS_LOOKUP_HIT_TOTAL = Counter(
    "vllm_llmd_fs_lookup_hit_total",
    "Total number of blocks found in the fs backend manager lookup.",
)

LLMD_FS_LOOKUP_TOTAL_BLOCKS = Counter(
    "vllm_llmd_fs_lookup_total_blocks",
    "Total number of blocks requested in the fs backend manager lookup.",
)
