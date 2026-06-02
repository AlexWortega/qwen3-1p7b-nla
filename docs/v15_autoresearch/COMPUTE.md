# COMPUTE
provider = local-ssh (eva01)
host = eva01   # 4×V100-SXM2-32GB, sm_70, no vLLM, fp16; Docker compose, -v /mnt/storage/vae_llm/artifacts:/big
parallelism = 3   # 3 of 4 GPUs (leave 1 for shared use / teacher)
notes = run via: ssh eva01 "cd ~/vae_llm && docker compose -f docker/compose.yml run --rm -e CUDA_VISIBLE_DEVICES=<g> -v /mnt/storage/vae_llm/artifacts:/big nla python <cmd>"
