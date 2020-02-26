# 로컬 이미지로 데이터셋 만들기
import os, re, glob
import cv2  # openCV 라이브러리 import하기
import numpy as np
from sklearn.model_selection import train_test_split

# 현재 로컬 이미지 폴더 구조

# dataset/25/road, water, building, green
imagePath = './dataset/dogs-vs-cats'
categories = ["dogs", "cats"]

# dataset/25 하위 폴더의 이름이 카테고리가 됨. 동일하게 맞춰줘야한다.
nb_classes = len(categories)

image_w = 28
image_h = 28

X = []
Y = []

for idx, cate in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = imagePath + '/' + cate + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            X.append(img / 256)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

# 생성된 데이터셋을 저장할 경로와 파일이름 지정
np.save("./train.npy", xy)