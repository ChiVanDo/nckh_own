import cvzone
import numpy as np
import cv2
import math
import time
import serial
import threading
from Resnet18_Fire.checkFire import detectFire_cvd

data = None
ther = True
def read_from_serial(ser):
    global data
    while ther:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            print("Serial 1:", data)  # In dữ liệu từ serial      
def checkMaMau(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange_lower = np.array([0, 50, 50])     # Ngưỡng dưới của màu cam trong không gian màu HSV
    orange_upper = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv_image, orange_lower, orange_upper)
    cv2.imshow("mask", mask)
    # Tìm các điểm ảnh có màu cam
    orange_pixels = cv2.countNonZero(mask)
    # Kiểm tra xem có pixel màu cam hay không
    print(orange_pixels)
    if orange_pixels > 1000:
        print("cháy.")
    else:
        print("không cháy.")
def drawKc(img, x,y,w,h):
    cv2.line(img, (320,y+int(h/2)), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
    cv2.line(img, (x+int(w/2), 240), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
def handlerAndSendToSignal(image, x,y,w,h):
    goc_quay_Px, goc_quay_Py = 0, 0
    direction = "null"
    if x != 0 and y != 0:
        Px = kcx(x, y, w, h)
        Py = kcy(x, y, w, h)
        if(Py > 12 or Px > 12):
            print("PixelX", Px)
            print("PixelY", Py)
            if x > 320 and y > 240:  # Phải/Dưới
                direction = "00"
            elif x > 320 and y < 240:  # Phải/Trên
                direction = "01"
            elif x < 320 and y > 240:  # Trái/Dưới
                direction = "10"
            elif x < 320 and y < 240:  # Trái/Trên
                direction = "11"
            
            goc_quay_Px = int((Px / 3.2) * 10)
            goc_quay_Py = int((Py / 4.3) * 10)
                
    
        # elif Px < 12 and Py < 12:
        #     direction = "OK"
            
    drawKc(image, x, y, w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # ve hinh chu nhat quanh contours
    print(goc_quay_Px,":", goc_quay_Py ,":", direction)
    return str(goc_quay_Px), str(goc_quay_Py), direction
def drawxy(img):
    cv2.line(img, (320,0), (320, 480), (221, 222, 217), 1) # dọc  y
    cv2.line(img, (0,240), (640, 240), (221, 222, 217), 1) # ngang x
    
    cv2.putText(img, "x" , (620 , 230), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, "y" , (330 , 20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 1, cv2.LINE_AA)
def kcx(x,y,w,h):
    # Tọa độ của điểm 1
    #x1, y1 = 0, 240
    # Tọa độ của điểm 2
    #x2, y2 = 320, 240 
    # tọa độ điểm 1
    # x1,y1 = 320, y+int(h/2)
    # tọa độ điểm 2
    # x2,y2 = x+int(w/2), y+int(h/2)
    kc = math.sqrt(( (x+int(w/2) - 320) )**2 + ((y+int(h/2)) - (y+int(h/2)))**2)
   
    return kc
def kcy(x,y,w,h):  
    # Tọa độ của điểm 1
    #x1, y1 = 0, 240
    # Tọa độ của điểm 2
    #x2, y2 = 320, 240 
    # tọa độ điểm 1
    #tam_lua = (x+int(w/2), y+int(h/2))
    
    # x1,y1 = x+int(w/2), 240
    # tọa độ điểm 2
    # x2,y2 = x+int(w/2), y+int(h/2)
    #distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    kc = math.sqrt(((x+int(w/2)) - (x+int(w/2)))**2 + ((y+int(h/2)) - 240)**2)
   
    return kc
def convlution(img, kernal):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = img.shape
    mtkernal = np.random.rand(kernal, kernal)
    kq = np.zeros([h - kernal + 1, w - kernal + 1])
    
    for row in range(0, h - kernal + 1):
        for col in range(0, w - kernal + 1):
            kq[row, col] = np.sum(img[row: row + kernal, col:col + kernal] * mtkernal)
    
    img = cv2.cvtColor(kq, cv2.COLOR_GRAY2BGR)            
    return img


    #return x1,y1,w1,h1
def save(img):
    imgName = "check_img.jpg"
    output_path = f"{'E:/NCKH_python/img_check'}/{imgName}" # lưu ảnh vào đường link img với imgName là tên
    cv2.imwrite(output_path, img)
    print("{}writen!".format(imgName))  


def main():
    # Running real time from webcam
    ser = serial.Serial("COM3", 9600)
    thread1 = threading.Thread(target=read_from_serial, args=(ser,))
    thread1.start()
    
    cap = cv2.VideoCapture(1)
    fire_cascade = cv2.CascadeClassifier('E:/NCKH_python/fire_detection.xml')
    last_send_time = time.time()
    flagMain = 2

    global data
    
    path_img = 'E:/NCKH_python/img_check/check_img.jpg'
    path_model = 'E:/NCKH_python/Resnet18_Fire/fire_model.pth'
    detector = detectFire_cvd(path_model)
    
    margin = 50
    while True:
        if(data == "ok"):
            flagMain = 1
        elif data == "test":
            flagMain = 0
                    
        if(flagMain == 1 and data == "check"):
            ret,image = cap.read()
            drawxy(image)
            x,y,w,h = 0,0,0,0
            fire = fire_cascade.detectMultiScale(image, scaleFactor=1.07, minNeighbors=2)
            for x, y, w, h in fire:
                #cv2.rectangle(image, (x - 10, y - 10), (x + (w+10), y + (h+10)), (255, 0, 0), 1)
                pass
            current_time = time.time()
            if current_time - last_send_time >= 0.5 and  w != 0:  # 0.5 giây = 500ms
                x1 = x + w + 20
                y1 = y + h + 20
                if(w > 0 and h > 0):
                    image_cut = image[y-margin:y+h+margin, x-margin:x+w+margin] # cắt ảnh
                    #image_cut = cv2.addWeighted(image_cut, 1, image_cut, 0, -85) #giảm độ sáng
                    n, m = 0 , 0
                    if image_cut is None:
                        pass
                    else:
                        n = image_cut.shape[0]
                        m = image_cut.shape[1]
                    
                    check = None
                    if n > 0 and m > 0:
                        image_cut = cv2.resize(image_cut ,(224,224))
                        save(image_cut)
                        if(detector.predict_fire(path_img) == 0):
                            check = True
                        else:
                            check = False
                
                    PackageGocPx, PackageGocPy, PackageDirection = handlerAndSendToSignal(image, x,y,h,w)
                    str = PackageGocPx + "\n" + PackageGocPy + "," + PackageDirection + "."  
                    if(PackageDirection != "null"):
                        print(check)
                        if(data == "check" and check == True):
                            ser.write(str.encode())
                            print(str)
                                 
                last_send_time = current_time 
            
            cv2.imshow('image',image)
            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Close")
                break 
            
        elif flagMain == 0:
            pass
            
            
    cap.release()
    cv2.destroyAllWindows()

main()
ther = False
 
