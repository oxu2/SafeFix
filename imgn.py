import os
import requests
from PIL import Image
from io import BytesIO

# 1. Fill these in with your ImageNet credentials:
USERNAME   = "oxx220000"
ACCESS_KEY = "b9e9597451d38d28831c9ca26c994cba91d714ad"

# 2. The five synsets you asked for:
WNIDS = [
    "n03001627",  # chair
    "n03063599",  # coffee_mug
    "n02871525",  # bookcase
    "n03085013",  # electric_guitar
    "n03179701",  # desk
]

# 3. Download loop:
for wnid in WNIDS:
    # construct the query URL
    url = (
        "https://www.image-net.org/api/text/imagenet.synset.geturls"
        f"?username={USERNAME}"
        f"&accesskey={ACCESS_KEY}"
        f"&wnid={wnid}"
    )
    print(f"Fetching URL list for {wnid}…")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    urls = [u.strip() for u in resp.text.splitlines() if u.strip()]

    # make folder
    out_dir = os.path.join("imagenet_subset", wnid)
    os.makedirs(out_dir, exist_ok=True)

    # download each image, skipping failures
    for idx, img_url in enumerate(urls):
        try:
            r = requests.get(img_url, timeout=5)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(os.path.join(out_dir, f"{idx:06d}.jpg"))
        except Exception:
            continue

    print(f"Saved {len(os.listdir(out_dir))} images to {out_dir}")