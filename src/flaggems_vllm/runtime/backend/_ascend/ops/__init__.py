# Copyright 2026 FlagOS Contributors
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

from .chunk_kda import chunk_kda
from .FLA import chunk_gla, parallel_nsa, parallel_nsa_compression
from .flash_mla import flash_mla
from .flash_mla_with_kvcache import flash_mla_with_kvcache
from .flashmla_sparse import flash_mla_sparse_fwd

__all__ = [
    "chunk_gla",
    "chunk_kda",
    "flash_mla",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "parallel_nsa",
    "parallel_nsa_compression",
]

# Some FLA callers import from the FLA package or its implementation module.
# The backend registry loads this module after the generic FLA package, so
# update those public bindings to the Ascend implementations as well.
try:
    import importlib as _importlib

    import flaggems_vllm.ops.FLA as _fla

    _fla.chunk_gla = chunk_gla
    _fla.chunk_kda = chunk_kda
    _fla.parallel_nsa = parallel_nsa
    _fla.parallel_nsa_compression = parallel_nsa_compression

    _importlib.import_module("flaggems_vllm.ops.FLA.chunk_gla").chunk_gla = chunk_gla
    _importlib.import_module("flaggems_vllm.ops.FLA.chunk_kda").chunk_kda = chunk_kda
    _importlib.import_module("flaggems_vllm.ops.FLA.parallel_nsa").parallel_nsa = (
        parallel_nsa
    )
    _importlib.import_module(
        "flaggems_vllm.ops.FLA.parallel_nsa_compression"
    ).parallel_nsa_compression = parallel_nsa_compression
except Exception:
    # The backend module can also be imported in isolation during packaging.
    pass
