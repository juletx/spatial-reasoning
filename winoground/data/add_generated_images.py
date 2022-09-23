import json
import random

random.seed(1)

WINOGROUND_DIR = "/data/local-files/?d=data/images/"
STABLE_DIFFUSION_DIR = "/data/local-files/?d=images/stable_diffusion/"

with open('examples.json', 'r+', encoding="UTF-8") as f:
    data = json.load(f)
    for i, d in enumerate(data):
        d['image_0'] = f"{WINOGROUND_DIR}{d['image_0']}.png"
        d['image_1'] = f"{WINOGROUND_DIR}{d['image_1']}.png"
        c = random.choice([0, 1])
        d['generated_image_0'] = f"{STABLE_DIFFUSION_DIR}ex_{i}_cap_{c}_img_0.png"
        d['generated_image_1'] = f"{STABLE_DIFFUSION_DIR}ex_{i}_cap_{1 - c}_img_0.png"
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()
