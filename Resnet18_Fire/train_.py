import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, alexnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Chuẩn bị dữ liệu huấn luyện và kiểm tra
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Chỉnh kích thước ảnh nếu cần thiết
    transforms.RandomHorizontalFlip(), #lật ngang ngẫu nhiên
    transforms.RandomRotation(degrees = 10), #xoay ảnh ngẫu nhiên
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa dữ liệu
])

pth_root = 'E:/NCKH_python/Resnet18_Fire/root'
pth_test = 'E:/NCKH_python/Resnet18_Fire/test'
train_data = torchvision.datasets.ImageFolder(root= pth_root, transform=data_transforms)
test_data = torchvision.datasets.ImageFolder(root=pth_test, transform=data_transforms)
len_train = len(train_data)
len_test = len(test_data)

print(len_train)
print(len_test)
print(train_data.class_to_idx)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)


# # # # Xây dựng mô hình
model = resnet18(pretrained=True)
final_fc_layer = model.fc
input_size = final_fc_layer.in_features
model.fc = nn.Linear(input_size, 2)  # Số lớp là 2 
model = model.to(device)


# # # # # Huấn luyện mô hình
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

sl_epoch = 3
# # # Training loop 
for epoch in range(sl_epoch):  # Số epoch
    running_loss = 0.0
    model.train() 
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs) #cho model học dũ liệu
        loss = loss_fn(outputs, labels) #tính sai số dữ liệu
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{sl_epoch}], "
              f"Batch [{i + 1}/{len(train_loader)}], "
              f"Loss: {loss.item():.4f}")

    # In thông tin tổng hợp của loss sau mỗi epoch
    print(f"Epoch [{epoch + 1}/{sl_epoch}], "
          f"Total Loss: {running_loss / len(train_loader):.4f}")

print("Training finished!")
# Lưu trạng thái của mô hình sau khi huấn luyện
torch.save(model.state_dict(), 'fire_model.pth')



def checking_model():
    
    test_data = torchvision.datasets.ImageFolder(root=pth_test, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    # Load trạng thái của mô hình đã huấn luyện
    model.load_state_dict(torch.load('fire_model.pth'))

    # Đánh giá hiệu suất trên tập kiểm tra
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

checking_model()