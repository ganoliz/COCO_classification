from pycocotools.coco import COCO
import torch 
import os 
import matplotlib.pyplot as plt

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

def load_annotations(annotation_file):
    annotations = []
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_filename = parts[0]
            label = int(parts[1])  # Convert string label to integer
            annotations.append((img_filename, label))
    return annotations

def save_checkpoint(model, optimizer, epoch, filepath="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}")
    return


def load_checkpoint(model, optimizer, filepath="best_model.pth"):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    return start_epoch

def show_img_example(dataloader, classes):
  fig, ax = plt.subplots(1, 2, figsize=(10, 10))
  for i in range(2):
    image_tensor, labels = next(iter(dataloader))
    image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    label_id = labels[0].item()
    ax[i].imshow(image)
    ax[i].title.set_text(classes[label_id])
  plt.show()
  return
