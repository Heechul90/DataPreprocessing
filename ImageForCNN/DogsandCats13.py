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


train_path = './minc-2500-tiny/train'
test_path = './minc-2500-tiny/val'

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




# from gluoncv.utils import viz
# viz.plot_image(d[2][1])  # index 0 is image, 1 is label
# viz.plot_image(d[4567][0])
##########################################################
from mxnet.gluon.model_zoo import vision

net = vision.alexnet()
####################################################################






# train

net.collect_params().initialize(mx.init.Xavier(magnitude = 0), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .11})
# 오차 함수
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

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