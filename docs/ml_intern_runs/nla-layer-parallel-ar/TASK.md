# Layer-parallel AR for joint NLA RL on V100s

Continue `/Users/aleksandrnikolich/Desktop/vae_llm`. Bottleneck of the joint-RL
trainer on V100-32GB is peak memory: 3 Qwen3-1.7B instances loaded (AV + AV_init
+ AR) + activations + Adam state ≈ 27-32 GB. Already maxed `expandable_segments`
+ grad-checkpoint. Cannot fit `B=4 G=4 = 16 effective` (OOMs at AR forward).

User asks for **layer-parallel AR** — split the 19-layer truncated Qwen3 AR
across two GPUs so per-GPU memory headroom opens up for bigger batch /
group-size in the RL inner loop.

## Reframing — what's the real goal?

The headline ask is "layer-parallel" but the actual win is a smaller per-GPU
memory footprint for AR. Three implementation paths, ordered by complexity:

1. **Model-parallel by component** (≈4h work). AV trainable on GPU0; AV_init on
   GPU1 (frozen, fp16 only, no Adam); AR on GPU2. Each card holds one model.
   Activations move via `tensor.to("cuda:1")` between forward stages. Trivial
   PyTorch — no FSDP, no Megatron, no manual layer split. Likely gets the most
   headroom for least risk.
2. **Pipeline-parallel AR** (≈2 days). Split AR's 19 layers across 2 GPUs
   (~10/~9 each). Forward streams hidden states across PCIe; backward
   reverses. Need micro-batching to overlap. Use `torch.distributed.pipeline`
   or hand-rolled `nn.Sequential`-with-`.to(device_id)` per chunk.
3. **FSDP-shard AR** (≈1 day). Wrap AR with `FullyShardedDataParallel(ShardingStrategy.FULL_SHARD)`
   on 2 GPUs. Each GPU holds half of AR's params/grad/optim state. All-gather
   on forward, reduce-scatter on backward. Memory wins but small models like
   1.7B don't always benefit — comms overhead can exceed savings on 4× V100
   with PCIe (no NVLink).

## Assumptions

- eva01 box: 4× V100-SXM2-32GB, NVLink between pairs (need to verify topology
  with `nvidia-smi topo -m`), CUDA 12.1, PyTorch 2.4.1, transformers 4.46+.
- Container `vae_llm:latest` already has torch.distributed installed.
- Reference: existing `scripts/train_joint_rl_paper.py` does `to("cuda")` on
  every model — moves to single device implicitly.
- Single training process (not torchrun) is acceptable — can use multiple
  CUDA streams within one process.

## Unknowns

- Does the AR adapter's LoRA wrapper survive `model.to("cuda:N")` with
  PEFT? Need to test. PEFT injects adapters as modules; should follow.
- Backward through cross-device tensor: `output.to("cuda:0")` followed by
  `.backward()` should work but may need explicit retain_graph or autograd
  hint. PyTorch handles this since ~1.5.
- nvidia-smi topology — NVLink (50 GB/s) vs PCIe (16 GB/s) makes a 3x
  speed difference for cross-device transfers.

## Success criterion

After the change:
- Trainer fits **B=8 G=2 (or B=4 G=4) = 16 effective** without OOM.
- Step time on the new config no worse than 1.5× the current 12s/step (so
  the speedup from bigger batch isn't eaten by cross-device traffic).
- Final FVE_pipeline_meannorm on the same eval ≥ G's 0.362 (probably better
  due to bigger contrastive batch).
- Trainer still saves correct AV / AR adapter files.
