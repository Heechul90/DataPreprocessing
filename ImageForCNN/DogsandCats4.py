### Semantic segmentation


# 함수 및 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random
import gluoncv
import mxnet as mx

from mxnet import gluon, nd
from glob import glob


gluoncv.data.transforms.presets

# Load image
image = mx.image.imread('dataset/dogs-vs-cats/training-set/dogs/dog.1.jpg')
print('shape:', image.shape)
print('data type:', image.dtype)
print('minimum value:', image.min().asscalar())
print('maximum value:', image.max().asscalar())

# Visualize image
plt.imshow(image.asnumpy())

#Transform image
from mxnet.gluon.data.vision import transforms

transforms_fn = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

image = transforms_fn(image)
print('shape:', image.shape)
print('data type:', image.dtype)
print('minimum value:', image.min().asscalar())
print('maximum value:', image.max().asscalar())

# Batch image
image = image.expand_dims(0)  # 몇번째
print(image.shape)

# Load model
network = gluoncv.model_zoo.get_model('fcn_resnet50_ade', pretrained=True)

# Make prediction
output = network.demo(image)
print(output.shape)

output = output[0]
print(output.shape)

# Closer look. pixel slice
px_height, px_width = 200, 300
px_logit = output[:, px_height: px_width]
px_logit.shape

px_probability = mx.nd.softmax(px_logit)
px_rounded_probability = mx.nd.round(px_probability*100)/100
print(px_rounded_probability)

class_index = mx.nd.argmax(px_logit, axis=0)
class_index = class_index[0].astype('int').asscalar()
print(class_index)

from gluoncv.data.ade20k.segmentation import ADE20KSegmentation

class_label = ADE20KSegmentation.CLASSES[class_index]
print(class_label)

output_proba = mx.nd.softmax(output, axis=0)

output_heatmap = output_proba[127]
plt.imshow(output_heatmap.asnumpy())

# Visualize most likely class
prediction = mx.nd.argmax(output, 0).asnumpy()
print(prediction.shape)

from gluoncv.utils.viz import get_color_pallete

prediction_image = get_color_pallete(prediction, 'ade20k')
prediction_image

