# === 1A) Locate Correct Class Root Directory ===
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms,models
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import itertools
import json

root = Path("input/train")

if __name__ == '__main__':
    def is_image_file(p):
        return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def count_image_files_in_dir(d):
        count = 0
        for f in d.iterdir():
            if f.is_file() and is_image_file(f):
                count += 1
        return count

    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        subdirs = [dirpath / d for d in dirnames if not d.startswith(".")]
        if len(subdirs) >= 5:
            with_images = sum(1 for sd in subdirs if count_image_files_in_dir(sd) > 0)
            if with_images >= 5:
                candidates.append((dirpath, len(subdirs), with_images))

    if not candidates:
        raise RuntimeError("Could not auto-detect class root. Please inspect the dataset tree manually.")
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    class_root = candidates[0][0]
    print("Detected class root:", class_root)

    sample_classes = sorted([d.name for d in class_root.iterdir() if d.is_dir()])[:10]
    print("Sample class folders:", sample_classes)

    data_dir = str(class_root)
    print(data_dir)



    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64

    # Define the transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE[0], scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(0, shear=0.1, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # for MobileNetV2
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # for MobileNetV2
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root='input/val',transform=val_transforms)



    # DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Get number of classes and class names
    num_classes = len(train_dataset.classes)
    print("Number of classes:", num_classes)
    print("Sample classes:", train_dataset.classes[:10])


    device = 'cuda'
    model = models.vit_b_16(
    weights=models.ViT_B_16_Weights.IMAGENET1K_V1
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后 2 个 Transformer block
    for block in model.encoder.layers[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        num_classes
    ).to(device)

    # 5) Set a lower learning rate for fine-tuning
    FINE_TUNE_LR = 1e-4
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINE_TUNE_LR)

    # 6) Set up loss function
    criterion = nn.CrossEntropyLoss()

    # 7) Setup training loop with callbacks
    # PyTorch doesn't have built-in callbacks, so we'll implement Early Stopping and Model Checkpoint manually


    # Set up Early Stopping and Model Checkpointing

    best_val_loss = float('inf')
    early_stop_count = 0
    patience = 3  # 在验证集损失不提升的情况下，最多等待的轮数
    ckpt_path_ft = "mobilenetv2_finetune_best.pth"  # 模型保存路径

    # To store the metrics for plotting 
    train_losses = [] 
    train_accuracies = [] 
    val_losses = [] 
    val_accuracies = []

    # Training loop for frozen and fine-tuning
    for epoch in range(20):  
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Train phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()  # Set to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No gradient computation during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path_ft)  # Save best model
            print('当前最优模型已更新')
            early_stop_count = 0  # Reset early stop count
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print("Early stopping triggered!")
            break

        print(f"Epoch [{epoch + 1}/20], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


    # Plot Learning Curves
    def plot_metric(train_metrics, val_metrics, metric_name):
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(train_metrics) + 1), train_metrics, label=f'Train {metric_name}')
        plt.plot(range(1, len(val_metrics) + 1), val_metrics, label=f'Val {metric_name}')
        plt.title(f'{metric_name.title()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.title())
        plt.legend()
        plt.grid(True)
        plt.show()



    plot_metric(train_accuracies, val_accuracies, 'accuracy')
    plot_metric(train_losses, val_losses, 'loss')



    # Evaluation function for classification report and confusion matrix
    def evaluate_model(model, val_loader, device, target_names):
        model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        # Iterate through the validation set and collect predictions
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Classification Report
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

        # Macro metrics
        macro_p = precision_score(all_labels, all_preds, average='macro')
        macro_r = recall_score(all_labels, all_preds, average='macro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Macro Precision: {macro_p:.4f}  Macro Recall: {macro_r:.4f}  Macro F1: {macro_f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, target_names, normalize=True)

    # Function to plot confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=True):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    # Assuming 'val_loader' is your validation DataLoader and 'target_names' are the class labels
    #target_names = val_loader.dataset.dataset.classes # Assuming you have set this in the dataset
    target_names = [str(i) for i in range(61)]
    evaluate_model(model, val_loader, device, target_names)


    MODEL_PATH = "mobilenetv2_plantvillage_61cls.pth"  # PyTorch model format

    # Save the model state_dict (recommended in PyTorch)
    torch.save(model.state_dict(), MODEL_PATH)

    print("Saved:", MODEL_PATH,)