import torch 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from fourclass_datasets import CocoFourClassDataset
from util import load_annotations, load_checkpoint
from torchvision import models
import argparse
import torch.nn as nn

def show_prediction(model, dataloader, classes):
    model.eval()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = next(iter(dataloader))  
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)  
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  

    # Select the first image in the batch
    img = inputs[0].cpu().numpy().transpose(1, 2, 0)  
    label = labels[0].item()  
    probs = probabilities[0].cpu().numpy() 

    # Plot the image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True Label: {classes[label]}")

    # Plot class probabilities
    plt.subplot(1, 2, 2)
    plt.bar(classes, probs, color='blue')
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Predicted Probabilities")

    plt.show()

def get_args():
    parser = argparse.ArgumentParser("Visualize predictions")

    parser.add_argument("--ann_path", type=str, default="annotation_val.txt")
    parser.add_argument("--image_dir", type=str, default="coco/images/val2017/")
    parser.add_argument("--classes", type=list, default=['bird', 'airplane', 'train', 'boat'])
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/resnet50_bestmodel_45.pth")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    ANN_PATH = args.ann_path
    IMAGE_DIR = args.image_dir
    CLASSES = args.classes

    annotations = load_annotations(ANN_PATH)
    view_data = CocoFourClassDataset(image_dir=IMAGE_DIR, annotations=annotations)
    view_dataloader = DataLoader(view_data, batch_size=1, shuffle=True)
    resnet50 = models.resnet50(pretrained=True)

    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4)  # 4 classes: 'bird', 'airplane', 'train', 'truck'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    load_checkpoint(model, optimizer=optimizer, filepath=args.model_path)
    show_prediction(model, view_dataloader, classes=CLASSES)