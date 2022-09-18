import json

IMAGE_DIR = "/data/local-files/?d=images/stable_diffusion/"

with open('examples.json', 'r+') as f:
    data = json.load(f)
    for i, d in enumerate(data):
        d['images_caption_0'] = [f'{IMAGE_DIR}ex_{i}_cap_0_img_{j}.png' for j in range(9)]
        d['images_caption_1'] = [f'{IMAGE_DIR}ex_{i}_cap_1_img_{j}.png' for j in range(9)]
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()
