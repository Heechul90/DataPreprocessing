### Semantic segmentation


# 함수 및 라이브러리

from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)


from mxnet.image import color_normalize
from mxnet import image
train_augs = [image.ResizeAug(224),
              image.HorizontalFlipAug(0.5),  # flip the image horizontally
              image.BrightnessJitterAug(.3), # randomly change the brightness
              image.HueJitterAug(.1)]         # randomly change hue

test_augs = [image.ResizeAug(224)]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')



# Now we can create a data iterator with the augmentations we defined.

from mxnet.gluon.data.vision import ImageRecordDataset
train_rec = './data/train/dog.rec'
validation_rec = './data/validation/dog.rec'
trainIterator = ImageRecordDataset(filename=train_rec,
                                   transform=lambda X, y: transform(X, y, train_augs))

validationIterator = ImageRecordDataset(filename=validation_rec,
                                        transform=lambda X, y: transform(X, y, test_augs))