import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, class_map=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(coco_json_path) as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    if class_map is None:
        class_map = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    for img_id, img_info in tqdm(images.items()):
        img_w, img_h = img_info['width'], img_info['height']
        label_path = os.path.join(output_dir, Path(img_info['file_name']).stem + '.txt')

        lines = []
        for ann in annotations_by_image.get(img_id, []):
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            cls = class_map.get(ann['category_id'], ann['category_id'])
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

if __name__ == "__main__":
    convert_coco_to_yolo(
        coco_json_path="annotations.json",   # <-- CHANGE
        images_dir="images",               # <-- CHANGE
        output_dir="labels",               # <-- CHANGE
    )
