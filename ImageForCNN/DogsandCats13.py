### 개고양이 이미지 분류

# 함수, 모듈 준비
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random
import gluoncv
import mxnet as mx


from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet import gluon


ctx = mx.cpu()
########################################################################################################################
### seed값 설정


########################################################################################################################
### dataset 경로 및 사이즈
train_path = '../dataset/dogs-vs-cats2/train'
test_path = '../dataset/dogs-vs-cats2/test'

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
    return data, label

# def transformer(data, label):
#     data = mx.image.imresize(data, 224, 224)
#     data = mx.nd.transpose(data, (2, 0, 1))
#     data = data.astype(np.float32)
#     return data, label

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.datasets.ImageFolderDataset(train_path, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.datasets.ImageFolderDataset(test_path, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')


for d, l in train_data:
    break

print(d.shape, l.shape)

for da, la in test_data:
    break
print(da.shape, la.shape)
########################################################################################################################
### graph
# from gluoncv.utils import viz
# viz.plot_image(d[2][1])  # index 0 is image, 1 is label
# viz.plot_image(d[4567][0])

########################################################################################################################
### model
net = nn.HybridSequential()
with net.name_scope():
    net.add(
        #
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        #
        nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        #
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        #
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        #
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        #
        nn.Dense(1, activation='sigmoid'))
########################################################################################################################
### train

net.collect_params().initialize(mx.init.Normal(), ctx = ctx)      # sigma=0.01
net.collect_params().initialize(mx.init.Xavier(), ctx = ctx)      # rnd_type='uniform', factor_type='avg', magnitude=3
net.collect_params().initialize(mx.init.Orthogonal(), ctx = ctx)  # scale=1.414, rand_type='uniform'

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

# 오차 함수
sigmoidbinary = gluon.loss.SoftmaxCrossEntropyLoss()

# accuracy
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
            loss = sigmoidbinary(output, label)
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