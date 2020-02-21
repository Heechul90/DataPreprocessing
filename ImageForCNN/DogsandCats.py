### 함수 및 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random
import gluoncv
import mxnet as mx

from glob import glob


##############################################################################
### 사이즈, 경로 설정
ROW, COL = 96, 96
path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats2/training_set/'

### 훈련용 이미지 데이터 조정

# os모듈은 환경 변수나 디텍토리, 파일등의 OS 자원을 제어할 수 있게 해주는 모듈
# dog
dog_path = os.path.join(path, 'dogs/dog.*')

# glob - 디렉토리 안에 있는 파일들을 리스트로 만들어주는 모듈
dogs = []
for dog_img in glob(dog_path):
    dog = mx.image.imread(dog_img)
    dog = mx.image.imresize(dog, ROW, COL)
    dog = mx.nd.transpose(dog.astype('float32'), (2, 0, 1))
    dogs.append(dog)
len(dogs)
dogs[0].shape

# cat
cat_path = os.path.join(path, 'cats/cat.*')

cats = []
for cat_img in glob(cat_path):
    cat = mx.image.imread(cat_img)
    cat = mx.image.imresize(cat, ROW, COL)
    cat = mx.nd.transpose(cat.astype('float32'), (2, 0, 1))
    cats.append(cat)
len(cats)
cats[0].shape


### 라벨마들기
# enumerate - 반복문 사용시 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
# dog=1
y_dog = [1 for item in enumerate(dogs)]
len(y_dog)

# cat=0
y_cat = [0 for item in enumerate(cats)]
len(y_cat)

##########################################################
# 리스트의 형태를 ndarray로 바꿔줌
dogs = mx.nd.array(dogs)
cats = mx.nd.array(cats)

y_dog = mx.nd.array(y_dog)
y_cat = mx.nd.array(y_cat)


### concatenate
X_train =  mx.nd.concatenate([dogs, cats], axis=0)


### append
# X
X_train = dogs


for i in cats:
    X_train.append(i)

len(X_train)

# y
y_train = y_dog

for i in y_cat:
    y_train.append(i)

len(y_train)

dogs[0:2]
a = X_train[0:10]
b = y_train[0:10]
type(a)
type(b)

df = pd.DataFrame({'Image': a,
                   'DogCat': b})
df[0]
len(df)

df.to_csv('dataset/trainset.csv',
          sep = ',')

data_set = pd.read_csv('dataset/trainset.csv', index_col=0)
data_set
d = data_set.values
d[0]
### concatenate
len(mx.nd.concatenate([a, b], axis=1))



##############################################################################
X_train
y_train
##############################################################################
a = X_train
list(a)
type(a)

from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)

ctx = mx.cpu()

# def transformer(data, label):
#     data = mx.image.imresize(data, 224, 224)
#     data = mx.nd.transpose(data, (2, 0, 1))
#     data = data.astype(np.float32)
#     return data, label
#
# batch_size = 64
# train_data = gluon.data.DataLoader(
#     gluon.data.vision.CIFAR10('./data', train = True, transform = transformer),
#     batch_size = batch_size, shuffle = False, last_batch = 'discard')
#
# test_data = gluon.data.DataLoader(
#     gluon.data.vision.CIFAR10('./data', train = False, transform = transformer),
#     batch_size = batch_size, shuffle = True, last_batch = 'discard')
#
#
# for d, l in train_data:
#     break
#
# print(d.shape, l.shape)
# train_data

# AlexNet
net = gluon.nn.Sequential()
        # 은닉층1 (채널=96, 커널=11, 패딩=1, 스트라이드=4, 활성화함수=relu)
        # maxpooling(사이즈=3, 스트라이드2)
        # 입력사이즈 (224, 224), 출력사이즈 (27, 27)
net.add(gluon.nn.Conv2D(96, kernel_size=11, padding=1, strides=4, activation='relu'))
net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))

        # 은닉층2 (채널=256, 커널=5, 패딩=2, 스트라이드=1, 활성화함수=relu)
        # maxpooling(사이즈=3, 스트라이드=2)
        # 입력사이즈(27, 27), 출력사이즈(13, 13)
net.add(gluon.nn.Conv2D(256, kernel_size=5, padding=2, strides=1, activation='relu'))
net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))

        # 은닉층3 (채널=384, 커널=3, 패딩=1, 스트라이드=1, 활성화함수=relu)
        # 입력사이즈(13, 13), 출력사이즈(13, 13)
net.add(gluon.nn.Conv2D(384, kernel_size=3, padding=1, strides=1, activation='relu'))

        # 은닉층4 (채널=384, 커널=3, 패딩=1, 스트라이드=1, 활성화함수=relu)
        # 입력사이즈(13, 13), 출력사이즈(13, 13)
net.add(gluon.nn.Conv2D(384, kernel_size=3, padding=1, strides=1, activation='relu'))

        # 은닉층5 (채널=256, 커널=3, 패딩=1, 스트라이드=1, 활성화함수=relu)
        # maxpooling(사이즈=3, 스트라이드=2)
        # 입력사이즈(13, 13), 출력사이즈(6, 6)
net.add(gluon.nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))

        # 1차원 배열로
gluon.nn.Flatten()

        # 은닉층6 (채널=4096, 활성화함수=relu)
        # dropout(0.5)
        # 입력사이즈(6, 6), 출력사이즈(1, 1)
net.add(gluon.nn.Dense(4096, activation="relu"), gluon.nn.Dropout(0.5))

        # 은닉층7 (채널=4096, 활성화함수=relu)
        # dropout(0.5)
        # 입력사이즈(1, 1), 출력사이즈(1, 1)
net.add(gluon.nn.Dense(4096, activation="relu"), gluon.nn.Dropout(0.5),)
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
net.add(gluon.nn.Dense(10))





net.collect_params().initialize(mx.init.Xavier(magnitude = 0), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .11})
# 오차 함수
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# def evaluate_accuracy(X_train, y_train, net):
#     acc = mx.metric.Accuracy()
#     for d, l in X_train, y_train:
#         data = d.as_in_context(ctx)
#         label = l.as_in_context(ctx)
#         output = net(data)
#         predictions = nd.argmax(output, axis = 1)
#         acc.update(preds = predictions, labels = label)
#     return acc.get()

#######
# Only one epoch so tests can run quickly, increase this variable to actually run
#######

epochs = 1
smoothing_constant = 0.01

for e in range(epochs):
    for i in range(len(X_train)):
        data = X_train.as_in_context(ctx)
        label = y_train.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ############
        # keep a moving average of the losses
        ############

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    # test_accuracy = evaluate_accuracy(test_data, net)
    # train_accuracy = evaluate_accuracy(train_data, net)
    # print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss))



##############################################################################
