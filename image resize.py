import cv2
import os 
import os, glob, numpy as np

folderPath = r"C:\Users\gunho\Desktop\Keras\bricks\Camera"
folderOutPath = r"C:\Users\gunho\Desktop\Keras\bricks\Camera_Out"

imageList = glob.glob("C:/Users/gunho/Desktop/python/brick/*/*.*")
print(imageList)
for i in imageList:
    # print(i)
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_AREA)
    ret, dst = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    e = i.replace("brick", "brick-Out").split("\\")
    e = e[0] + "/" + e[1]
    if not os.path.exists(e):
        print(e)
        # e.remove(e[2])
        os.mkdir(e)
    cv2.imwrite(i.replace("brick", "brick-Out"), dst)