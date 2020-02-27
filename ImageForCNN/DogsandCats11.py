### 개고양이 이미지 분류

# 함수, 모듈 준비
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random
import gluoncv
import mxnet as mx

from mxnet import nd, autograd
from mxnet import gluon
from glob import glob
from keras.preprocessing import image

ctx = mx.cpu()
##########################################################
### seed값 설정
# seed 값은 random 함수에서 랜덤 값을 계산할 때 사용하며 매 번 바뀝니다.
# 초기 seed 값을 설정하지 않으면 랜덤 값을 생성하는 순서가 매 번 달라집니다.
# seed = 0
# np.random.seed(seed)
# tf.set_random_seed(seed)

##########################################################
### dataset 경로 및 사이즈
ROW, COL = 96, 96
path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train1/'

##########################################################
### 데이터 불러오기(dog_train)
# os - 환경 변수나 디렉토리, 파일 등의 OS자원을 제어할 수 있게 해주는 모듈
# 함수	                설명
# os.mkdir(디렉터리)	    디렉터리를 생성한다.
# os.rmdir(디렉터리)	    디렉터리를 삭제한다.단, 디렉터리가 비어있어야 삭제가 가능하다.
# os.unlink(파일)	    파일을 지운다.
# os.rename(src, dst)	src라는 이름의 파일을 dst라는 이름으로 바꾼다.

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
dog_path = os.path.join(path, 'dog.*')
len(glob(dog_path))

dogs = []
for dog_image in glob(dog_path):
    dog = cv2.imread(dog_image)                           # 이미지 데이터를 읽기
    dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    dog = cv2.resize(dog, (ROW, COL))                     # 가로,세로 사이즈 설정
    dog = image.img_to_array(dog)                         # 이미지를 array로 변환
    dog = np.transpose(dog, (2, 0, 1))
    dogs.append(dog)
len(dogs)

##########################################################
# 데이터 불러오기(cat_train)
cat_path = os.path.join(path, 'cat.*')
len(glob(cat_path))

cats = []
for cat_image in glob(cat_path):
    cat = cv2.imread(cat_image)
    cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
    cat = cv2.resize(cat, (ROW, COL))
    cat = image.img_to_array(cat)
    cat = np.transpose(cat, (2, 0, 1))
    cats.append(cat)
len(cats)

##########################################################
# enumerate - 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
# dog=0, cat=1
y_dog, y_cat = [], []

y_dog = [0 for item in enumerate(dogs)]
y_cat = [1 for item in enumerate(cats)]

len(y_dog)
len(y_cat)


##########################################################
# 리스트의 형태를 ndarray로 바꿔줌
dogs = mx.nd.array(dogs).astype('float32')
cats = mx.nd.array(cats).astype('float32')

y_dog = mx.nd.array(y_dog).astype('int32')
y_cat = mx.nd.array(y_cat).astype('int32')

##########################################################
# 표준화를 하면 학습 속도가 더 빨라지고, 지역 최적의 상태에 빠지게 될 가능성을 줄여준다
# local optima 요즘 trend에 의하면, 중요한 문제가 아니다(실제 딥러닝에서 local optima에 빠질 확률이 거의 없음)
# values값을 0과 1 사이로 맞춰줌
dogs /= 255                                                # 이미지는 숫자로 0~255의 8비트 부호없는 정수로 저장
cats /= 255


##########################################################
# concatenate 함수를 이용해서 배열 결합
X = mx.nd.concatenate([dogs, cats], axis=0)
y = mx.nd.concatenate([y_dog, y_cat], axis=0)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.3)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

train_data = mx.io.NDArrayIter(data=X_train, label=Y_train)

test_data = mx.io.NDArrayIter(data=X_test, label=Y_test)


##########################################################



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

for d, l in train_data:
    print (d)
    print (l)



def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis = 1)
        acc.update(preds = predictions, labels = label)
    return acc.get()

#######
# Only one epoch so tests can run quickly, increase this variable to actually run
#######

epochs = 1
smoothing_constant = 0.01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
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

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))