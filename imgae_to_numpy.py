import cv2
import os 
from PIL import Image
import glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

imageArray = []

folderPath = r"C:\Users\gunho\Desktop\Keras\bricks\Camera"
folderOutPath = r"C:\Users\gunho\Desktop\python\brick-Out"

imageList = glob.glob(folderOutPath+"/*/*.*")

# print(imageList)
image_w = 64
image_h = 64

X = []
Y = []

categories = os.listdir(folderOutPath)
print(categories)
for idx, cat in enumerate(categories):
    label = []
    for i in categories:
        label.append(0)
    label[idx] = 1

    for i, f in enumerate(imageList):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        # img.show()
        data = np.asarray(img)

        X.append(data)
        Y.append(label)
X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./binary_image_data.npy", xy)
print(X)