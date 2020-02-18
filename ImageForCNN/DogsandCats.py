### 함수 및 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random
import gluoncv
import mxnet as mx

from glob import glob
from keras.preprocessing import image

### 사이즈, 경로 설정
ROW, COL = 224, 224
path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'

### 훈련용 이미지 데이터 조정
# dog
dog_path = os.path.join(path, 'dog.*')

dogs = []
for dog_img in glob(dog_path):
    dog = mx.image.imread(dog_img)
    dog = mx.image.imresize(dog, ROW, COL)
    dog = mx.nd.transpose(dog.astype('float32'), (2, 0, 1)) / 255
    dogs.append(dog)



# cat
cat_path = os.path.join(path, 'cat.*')

cats = []
for cat_img in glob(cat_path):
    cat = mx.image.imread(cat_img)
    cat = mx.image.imresize(cat, ROW, COL)
    cat = mx.nd.transpose(cat.astype('float32'), (2, 0, 1)) / 255
    cats.append(cat)


### 라벨마들기
# dog=1
y_dog = [1 for item in enumerate(dogs)]
len(y_dog)

# cat=0
y_cat = [0 for item in enumerate(dogs)]
len(y_cat)

### concatenete
a = dogs[0]
b = cats[0]


X_train = mx.nd.concatenate([dogs], [cats])
X_train[0]

