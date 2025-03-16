
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
from fourclass_datasets import CocoFourClassDataset
from util import save_checkpoint, load_annotations
import argparse

def train(train_ann, train_image_dir, val_ann, val_image_dir, num_epochs, batch_size, lr, eval_interval, save_path):
    resnet50 = models.resnet50(pretrained=True)

    for param in resnet50.parameters():
        param.requires_grad = False

    for param in resnet50.layer3.parameters():
        param.requires_grad = True

    for param in resnet50.layer4.parameters():
        param.requires_grad = True


    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4)  # 4 classes: 'bird', 'airplane', 'train', 'truck'
    )

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    annotations_train = load_annotations(train_ann)
    annotations_val = load_annotations(val_ann)
    training_data = CocoFourClassDataset(image_dir=train_image_dir, annotations=annotations_train,
                        transform=train_transforms)
    val_data = CocoFourClassDataset(image_dir=val_image_dir, annotations=annotations_val,
                        transform=val_transforms)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    num_epochs = num_epochs
    best_val_accuracy = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataloader.dataset)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
            scheduler.step(epoch_loss)


            if epoch % eval_interval == 0:
                model.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                    val_accuracy = correct / total
                    print(f"Validation Accuracy after {epoch} epochs: {val_accuracy * 100:.2f}%")

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        save_checkpoint(model, optimizer, epoch, filepath=f"{save_path}resnet50_best_{epoch}.pth")

def get_args():
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--ann_path", type=str, default="annotation_train.txt")
    parser.add_argument("--image_dir", type=str, default="coco/images/train2017/")
    parser.add_argument("--val_ann_path", type=str, default="annotation_val.txt")
    parser.add_argument("--val_image_dir", type=str, default="coco/images/val2017/")
    parser.add_argument("--num_epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("-eval-interval", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="/content/drive/MyDrive/")
    
    return parser.parse_args()

if __name__ == '__main__':
    parser = get_args()
    ANN_PATH = parser.ann_path
    IMAGE_DIR = parser.image_dir
    VAL_ANN_PATH = parser.val_ann_path
    VAL_IMAGE_DIR = parser.val_image_dir
    NUM_EPOCHS = parser.num_epochs
    BATCH_SIZE = parser.batch_size
    LR = parser.lr
    EVAL_INTERVAL = parser.eval_interval
    MODEL_PATH = parser.save_path
    train(ANN_PATH, IMAGE_DIR, VAL_ANN_PATH, VAL_IMAGE_DIR,
           NUM_EPOCHS, BATCH_SIZE, LR, EVAL_INTERVAL, MODEL_PATH)