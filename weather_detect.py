# ==========================
# Cold vs Hot Scene Detector
# ==========================

import cv2

img = cv2.imread("scene.jpg")
blue = img[:,:,0].mean()
red = img[:,:,2].mean()

print("Cold Scene" if blue>red else "Hot Scene ")
