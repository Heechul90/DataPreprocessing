### 함수 및 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random
import gluoncv
import mxnet as mx

from glob import glob

### 사이즈, 경로 설정
ROW, COL = 224, 224
path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train1/'

### 훈련용 이미지 데이터 조정
# dog
dog_path = os.path.join(path, 'dog.*')

dogs = []
for dog_img in glob(dog_path):
    dog = cv2.imread(dog_img)
    dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
    dog = cv2.resize(dog, (ROW, COL))
    dogs.append(dog)
len(dogs)


# cat
cat_path = os.path.join(path, 'cat.*')

cats = []
for cat_img in glob(cat_path):
    cat = cv2.imread(cat_img)
    cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
    cat = cv2.resize(cat, (ROW, COL))
    cat = image.img_to_array(cat)
    cats.append(cat)
len(cats)

### 사진 확인하기



### 라벨마들기
# dog=1
y_dog = [1 for item in enumerate(dogs)]
len(y_dog)

# cat=0
y_cat = [0 for item in enumerate(dogs)]
len(y_cat)

## converting everything to Numpy array to fit in our model
## them creating a X and target file like we used to see
## in Machine and Deep Learning models
dogs = np.asarray(dogs).astype('float32')
cats = np.asarray(cats).astype('float32')
y_dog = np.asarray(y_dog).astype('int32')
y_cat = np.asarray(y_cat).astype('int32')

### 값을 0과 1 사이의 값으로 바꾸기
dogs = dogs / 255

### concatenete
X_train = np.concatenate((dogs, cats), axis = 0)
X_train[0]