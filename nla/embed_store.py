"""Single-slot ray actor mailbox for the rollout embedding table.

Megatron actor rank-0 calls set() after each weight sync; the DRIVER calls
get() and pushes the tensor to RolloutManager via set_embed. Passing the 2GB
tensor through ray actor RPC keeps borrow-tracking intact (torch.save of an
ObjectRef → torch.load → ray.get does NOT — out-of-band ref, ray.get hangs).

This avoids any NFS I/O on the embed path. The original ~33min/save stall was
the rank-0 NFS-ops-before-barrier deadlock in save_model (not embed I/O), but
NFS reads from RolloutManager's asyncio loop still hung under bg-write
saturation, so the embed must not touch NFS either.

The store holds exactly one tensor; set() overwrites → prior GC'd.

FSDP doesn't use this — it writes the full tensor to /dev/shm (single-node,
actor and RolloutManager colocated).
"""

import ray
import torch


_NAME = "_nla_embed_store"


@ray.remote(num_cpus=0)
class _NLAEmbedStore:
    def __init__(self):
        self._weight: torch.Tensor | None = None

    def set(self, weight: torch.Tensor) -> None:
        self._weight = weight

    def get(self) -> torch.Tensor | None:
        return self._weight


def get_embed_store():
    return _NLAEmbedStore.options(
        name=_NAME, namespace="nla", lifetime="detached", get_if_exists=True
    ).remote()
