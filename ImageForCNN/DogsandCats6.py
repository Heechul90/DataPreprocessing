import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

train_dir = 'D:/HeechulFromGithub/DataPreprocessing(mxnet)/dataset/dogs-vs-cats2/train'
test_dir = 'D:/HeechulFromGithub/DataPreprocessing(mxnet)/dataset/dogs-vs-cats2/test'
img_size = 50
lr = 1e-3

model_name = 'dogsvscats-{}-{}.model'.format(lr, '2conv-basic')

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat' : return [1,0]
    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


