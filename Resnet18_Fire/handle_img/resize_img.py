import cv2
import os
def convert_to_grayscale(input_image_path, output_image_path):
    # Đọc ảnh màu từ đường dẫn
    img = cv2.imread(input_image_path)
    # Chuyển đổi ảnh màu thành ảnh đen trắng
    
    img = cv2.resize(img, (512,512))
    # Lưu ảnh đen trắng
    cv2.imwrite(output_image_path, img)
    print(f"Đã lưu ảnh đen trắng tại: {output_image_path}")
    
# Đường dẫn của ảnh màu đầu vào và đầu ra



for i in range(1, 36):
    input_image = f"\no_fire\{i}.jpg"
    output_image = f"\no_fire\{i}.jpg"
    # Gọi hàm để chuyển đổi và lưu ảnh đen trắng
    convert_to_grayscale(input_image, output_image)