import cv2
import math
import time
import serial
import threading
from Resnet18_Fire.checkFire import detectFire_cvd
import os
from PIL import Image

def save(img):
    imgName = "check_img.jpg"
    output_path = f"{'E:/NCKH_python/img_check'}/{imgName}" # lưu ảnh vào đường link img với imgName là tên
    cv2.imwrite(output_path, img)
    print("{}writen!".format(imgName))  


def main():
    cap = cv2.VideoCapture(1)
    path_model = 'E:/NCKH_python/Resnet18_Fire/fire_model.pth'
    detector = detectFire_cvd(path_model)
    last_send_time = time.time()
    path_img = 'E:/NCKH_python/img_check/check_img.jpg'
    fire_cascade = cv2.CascadeClassifier('E:/NCKH_python/fire_detection.xml')
    
    margin = 60
    while True:
        _,img = cap.read()
        
        x,y,w,h = 0,0,0,0
        fire = fire_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)
        for x, y, w, h in fire:
            pass
        
        current_time = time.time()
        if current_time - last_send_time >= 1 and w > 0 and h > 0:
            
            img_cut = img[y-margin:y+h+margin, x-margin:x+w+margin] # cắt ảnh
            n = img_cut.shape[0]
            m = img_cut.shape[1]
            
            print(n, m)
            if n > 0 and m > 0:
                img_cut = cv2.resize(img_cut, (512,512))
                save(img_cut)
                print(detector.predict_fire(path_img))
                
            last_send_time = current_time 
            
        
        
        cv2.imshow('image',img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Close")
            break 
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()