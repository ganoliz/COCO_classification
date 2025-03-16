from pycocotools.coco import COCO
import argparse

def get_single_category_images(coco, category_name):
    cat_id = coco.getCatIds(catNms=[category_name])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])

    filtered_img_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        all_cats = {ann['category_id'] for ann in anns}

        # Ensure only the desired category is present
        if len(all_cats) == 1 and cat_id in all_cats:
            filtered_img_ids.append(img_id)

    return filtered_img_ids

def get_args():
    parser = argparse.ArgumentParser("Prepare annotations")
    parser.add_argument("--ann_path", type=str, default="/content/coco/annotations/instances_train2017.json")
    parser.add_argument("--val_ann_path", type=str, default="/content/coco/annotations/instances_val2017.json")
    parser.add_argument("--classes", type=list, default=['bird', 'airplane', 'train', 'boat'])
    parser.add_argument("--out-ann-path", type=str, default="/content/annotation_train.txt")
    parser.add_argument("--out-val-ann-path", type=str, default="/content/annotation_val.txt")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    ANN_PATH = args.ann_path
    VAL_ANN_PATH = args.val_ann_path
    OUT_ANN_PATH = args.out_ann_path
    OUT_VAL_ANN_PATH = args.out_val_ann_path
    classes = args.classes

    coco = COCO(ANN_PATH)
    with open(OUT_ANN_PATH, "w+") as f:
        for coco_class in classes:
            # img_ids = coco.getImgIds(catIds=coco.getCatIds(catNms=coco_class))
            img_ids = get_single_category_images(coco, coco_class)
            for i in range(len(img_ids)):
                image_name = str(img_ids[i]).rjust(12, '0')
                f.write(f"{image_name}.jpg {classes.index(coco_class)}\n")

    coco = COCO(VAL_ANN_PATH)
    with open(OUT_VAL_ANN_PATH, "w+") as f:
        for coco_class in classes:
            img_ids = get_single_category_images(coco, coco_class)
            for i in range(len(img_ids)):
                image_name = str(img_ids[i]).rjust(12, '0')
                f.write(f"{image_name}.jpg {classes.index(coco_class)}\n")