import os
import cv2
import numpy as np
# Đường dẫn đến thư mục chứa các ảnh
folder_path = 'E:/NCKH_python/Resnet18_Fire/root/no_fire'
out_path_xoay = 'E:/NCKH_python/Resnet18_Fire/root/no'
count = 1
# Lặp qua tất cả các tệp trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):  # Chỉ đọc các file ảnh cụ thể
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        # cv2.imshow('Original Image', image)
        if image is not None:
            # Ở đây, bạn có thể thực hiện các thao tác xử lý ảnh với biến 'image'
            image_xoay = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # Tạo ma trận nhiễu salt
            # # Tạo nhiễu ngẫu nhiên cho từng kênh màu
            # salt_noise = np.random.randint(0, 30, size=image.shape[:2])  # Điều chỉnh mức độ nhiễu tại đây
            # # Thêm nhiễu salt vào ảnh
            # image[salt_noise == 0] = (255, 255, 255)  # Điều chỉnh giá trị nhiễu để phù hợp với độ sáng
            
            #nhiễu cả 3 kênh
            noise = np.random.normal(0, 20, image.shape[:2])  # Điều chỉnh mức độ nhiễu tại đây
            # Thêm nhiễu vào từng kênh màu của ảnh
            image[:, :, 0] = np.where((image[:, :, 0] + noise) > 0, np.uint8(np.minimum(255, image[:, :, 0] + noise)), 0)
            image[:, :, 1] = np.where((image[:, :, 1] + noise) > 0, np.uint8(np.minimum(255, image[:, :, 1] + noise)), 0)
            image[:, :, 2] = np.where((image[:, :, 2] + noise) > 0, np.uint8(np.minimum(255, image[:, :, 2] + noise)), 0)

            # Hiển thị ảnh gốc
            # Hiển thị ảnh sau khi thêm nhiễu
            # cv2.imshow('Noisy Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
             # Đặt tên cho tệp ảnh đã xoay
            # new_filename = f'img_xoay_{count}.jpg'
            # # Đường dẫn để lưu ảnh đã xoay
            # output_path = os.path.join(out_path_xoay, new_filename)
            cv2.imwrite(f"E:/NCKH_python/Resnet18_Fire/root/no_fire/imgnhieu_2{count}.jpg", image)
            # Lưu ảnh đen trắng
            print(f"Đã lưu ảnh xoay tại:", count)
            count = count + 1
           
        
            # Ví dụ: hiển thị ảnh
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
