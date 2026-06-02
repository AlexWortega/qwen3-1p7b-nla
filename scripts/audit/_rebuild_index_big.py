"""Rebuild activations_pool_v9/index.json from all *.meta.json shards. Needed because
extract_multi.py overwrites index.json with only the tags in its run config — re-run this
after any single-tag extraction to restore the full pool index. Run inside docker (root)."""
import json, glob, os
d = "/big/activations_pool_v9"
idx = {}
for m in sorted(glob.glob(os.path.join(d, "*.meta.json"))):
    tag = os.path.basename(m)[:-len(".meta.json")]
    meta = json.load(open(m))
    shard = os.path.join(d, meta.get("shard", tag + ".safetensors"))
    if os.path.exists(shard):
        idx[tag] = meta
json.dump(idx, open(os.path.join(d, "index.json"), "w"), indent=2, sort_keys=True)
print("rebuilt index:", len(idx), "tags:", sorted(idx))
