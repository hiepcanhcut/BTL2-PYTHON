import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Định nghĩa CNN với BatchNorm, Dropout, và nhiều lớp hơn
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(512)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        # Sau 4 lần pooling, kích thước ảnh CIFAR-10 (32x32) sẽ còn 2x2 với 512 channel
        self.fc      = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool(nn.functional.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def main():
    # 2. Chuẩn bị dữ liệu với augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_len = int(0.9 * len(full_train))
    val_len   = len(full_train) - train_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])
    # Với validation, không dùng augmentation, chỉ normalize
    val_ds.dataset.transform = transform_test

    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # DataLoader: đặt num_workers=0 để tránh lỗi (hoặc có thể để num_workers=2 sau khi bọc if __name__=='__main__')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

    # 3. Khởi tạo model, criterion, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Scheduler: giảm lr mỗi 10 epoch với factor 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 25
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    # 4. Train & Validate loop
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        tloss, tcorrect = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            tloss += loss.item() * X.size(0)
            tcorrect += (out.argmax(1) == y).sum().item()

        train_loss = tloss / len(train_loader.dataset)
        train_acc  = tcorrect / len(train_loader.dataset)

        # Validate
        model.eval()
        vloss, vcorrect = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                vloss += criterion(out, y).item() * X.size(0)
                vcorrect += (out.argmax(1) == y).sum().item()

        val_loss = vloss / len(val_loader.dataset)
        val_acc  = vcorrect / len(val_loader.dataset)

        # Cập nhật history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)

        # In kết quả từng epoch
        print(f"Epoch {epoch}/{epochs}  —  "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Step scheduler vào cuối mỗi epoch
        scheduler.step()

    # 5. Vẽ đồ thị Loss & Accuracy (có thể comment nếu không muốn hiển thị)
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Validation Loss')
    plt.title('ImprovedCNN - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'],   label='Validation Accuracy')
    plt.title('ImprovedCNN - Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Đánh giá trên test set và vẽ Confusion Matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            out = model(X)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=test_ds.classes)
    plt.figure(figsize=(6,6))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('ImprovedCNN - Confusion Matrix')
    plt.tight_layout()
    plt.show()

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == '__main__':
    main()
