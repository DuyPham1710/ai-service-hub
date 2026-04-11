import urllib.request
import json

# Search more broadly for good RVC models
searches = [
    "rvc v2 model",
    "rvc voice model",
    "rvc model pth",
    "rvc singing voice",
]

seen = set()

for query in searches:
    url = f"https://huggingface.co/api/models?search={query.replace(' ', '+')}&sort=likes&direction=-1&limit=20"
    req = urllib.request.Request(url)
    try:
        res = urllib.request.urlopen(req, timeout=10)
        data = json.loads(res.read())
        for m in data:
            mid = m["id"]
            if mid in seen:
                continue
            seen.add(mid)
            likes = m.get("likes", 0)
            
            # Check files
            tree_url = f"https://huggingface.co/api/models/{mid}/tree/main"
            try:
                tree_req = urllib.request.Request(tree_url)
                tree_res = urllib.request.urlopen(tree_req, timeout=5)
                files = json.loads(tree_res.read())
                pth_files = [f for f in files if f["path"].endswith(".pth")]
                idx_files = [f for f in files if f["path"].endswith(".index")]
                
                if pth_files:
                    for pf in pth_files:
                        size_mb = pf.get("size", 0) / 1024 / 1024
                        print(f"[{likes} likes] {mid}")
                        print(f"  .pth: {pf['path']} ({size_mb:.1f} MB)")
                    for idx in idx_files:
                        print(f"  .index: {idx['path']}")
                    print()
            except:
                pass
    except Exception as e:
        print(f"Search error for '{query}': {e}")
