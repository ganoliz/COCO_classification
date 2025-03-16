from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from fourclass_datasets import CocoFourClassDataset
from util import load_annotations, load_checkpoint
import argparse

def evaluate(model, dataloader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    print("Confusion Matrix:")
    print(conf_matrix)

    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for i, acc in enumerate(per_class_accuracy):
        print(f"{classes[i]} accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))



def get_args():
    parser = argparse.ArgumentParser("Evaluate model")

    parser.add_argument("--ann_path", type=str, default="annotation_val.txt")
    parser.add_argument("--image_dir", type=str, default="coco/images/val2017/")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/resnet50_bestmodel_45.pth")
    parser.add_argument("--classes", type=list, default=['bird', 'airplane', 'train', 'boat'])
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    ANN_PATH = args.ann_path
    IMAGE_DIR = args.image_dir
    MODEL_PATH = args.model_path
    CLASSES = args.classes

    annotations = load_annotations(ANN_PATH)
    eval_data = CocoFourClassDataset(image_dir=IMAGE_DIR, annotations=annotations)
    eval_dataloader = DataLoader(eval_data, batch_size=64, shuffle=False)

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4)  # 4 classes: 'bird', 'airplane', 'train', 'boat'
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    load_checkpoint(model, optimizer, MODEL_PATH)
    evaluate(model, eval_dataloader, CLASSES)
