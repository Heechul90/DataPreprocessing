from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)

ctx = mx.cpu()

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.datasets('D:\HeechulFromGithub\dataset\dogs-vs-cats\train', train = True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')


for d, l in train_data:
    break

print(d.shape, l.shape)