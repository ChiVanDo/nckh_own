from Resnet18_Fire.checkFire import detectFire_cvd
import cv2


path_model = 'E:/NCKH_python/Resnet18_Fire/fire_model.pth'
detector = detectFire_cvd(path_model)
path_img = 'E:/NCKH_python/Resnet18_Fire/root/fire/5.jpg'
pth_test = 'E:/NCKH_python/Resnet18_Fire/test'

print(detector.predict_fire(path_img))
print(detector.check_model(path_model, pth_test))