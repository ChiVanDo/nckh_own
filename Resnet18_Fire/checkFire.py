import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score

class detectFire_cvd:
    def __init__(self, path_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Khởi tạo mô hình ResNet-18
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 2)  # Số lớp là 2 
        self.model = self.model.to(self.device)
        # Tải trạng thái đã lưu của mô hình
        
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Chỉnh kích thước ảnh nếu cần thiết
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees = 10),
            transforms.ToTensor(),  # Chuyển ảnh thành tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa dữ liệu
        ])
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval()  #chế độ danh giá
     
        
        
    def predict_fire(self, path_img):
        image = Image.open(path_img)
        image = self.data_transforms(image).unsqueeze(0).to(self.device)
        # Dự đoán lớp của ảnh
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        # In ra kết quả dự đoán
        return predicted.item()
    


    def check_model(self, path_model, path_data_test):
        test_data = ImageFolder(root=path_data_test, transform=self.data_transforms)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
        # Load trạng thái của mô hình đã huấn luyện
        self.model.load_state_dict(torch.load(path_model))

        # Đánh giá hiệu suất trên tập kiểm tra
        self.model.eval()
        predic = []
        y_test = []
        with torch.no_grad(): # tắt gradient giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán.
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
            
                y_test.extend(labels.tolist())  # Chuyển tensor sang list và thêm vào mảng
            
                # Gán kết quả dự đoán vào mảng predicted_labels
                predic.extend(predicted.tolist())  # Chuyển t
        print(len(predic), len(y_test))
        return (100* accuracy_score(y_test,predic))





# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Khởi tạo mô hình ResNet-18
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(512, 2)  # Số lớp là 2 
# model = model.to(device)

# # Tải trạng thái đã lưu của mô hình
# model.load_state_dict(torch.load('fire_model.pth'))
# model.eval()  # Chuyển sang chế độ đánh giá (không huấn luyện)


# # Đường dẫn đến ảnh bạn muốn dự đoán
# image_path = '26.jpg'

# # Load và tiền xử lý ảnh
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),  # Chỉnh kích thước ảnh nếu cần thiết
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(degrees = 10),
#     transforms.ToTensor(),  # Chuyển ảnh thành tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa dữ liệu
# ])



# def checking_model():
#     test_data = ImageFolder(root='test', transform=data_transforms)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
#     # Load trạng thái của mô hình đã huấn luyện
#     model.load_state_dict(torch.load('fire_model_final.pth'))

#     # Đánh giá hiệu suất trên tập kiểm tra
#     model.eval()
#     predic = []
#     y_test = []
#     with torch.no_grad(): # tắt gradient giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán.
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
        
#             y_test.extend(labels.tolist())  # Chuyển tensor sang list và thêm vào mảng
        
#             # Gán kết quả dự đoán vào mảng predicted_labels
#             predic.extend(predicted.tolist())  # Chuyển t
#     print(len(predic), len(y_test))
#     print ("Tỷ lệ đúng khi dự đoán mô hình: %.2f %%" %(100* accuracy_score(y_test,predic)))
    # print(f'Accuracy on test set: {accuracy:.2f}%')
#checking_model()