from fourclass_datasets import CocoFourClassDataset
from torch.utils.data import DataLoader
from util import show_img_example, load_annotations
import torchvision.transforms as transforms
import os
import argparse


def visualize(ann_path, image_dir, classes):
    view_transforms = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor()])
    annotations = load_annotations(ann_path)
    view_data = CocoFourClassDataset(image_dir=image_dir, annotations=annotations,transform=view_transforms)

    view_dataloader = DataLoader(view_data, batch_size=2, shuffle=True)

    show_img_example(view_dataloader, classes=classes)

def get_args():
    parser = argparse.ArgumentParser("Visualize dataset")

    parser.add_argument("--ann_path", type=str, default="annotation_train.txt")
    parser.add_argument("--image_dir", type=str, default="coco/images/train2017/")
    parser.add_argument("--classes", type=list, default=['bird', 'airplane', 'train', 'boat'])

    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()
    ANN_PATH = args.ann_path
    IMAGE_DIR = args.image_dir
    CLASSES = args.classes
    visualize(ANN_PATH, IMAGE_DIR, CLASSES)