import numpy as np
import cv2
import time
import subprocess
import scipy as sp
import glob, os
import mxnet as mx
import pickle
import random
from pandas import *
from sklearn import preprocessing
from PIL import Image
from resizeimage import resizeimage

f_path = "D:/HeechulFromGithub/DataPreprocessing(mxnet)/dataset/dogs-vs-cats/training-set/"

img_name = "IMG_17"

filelist = glob.glob(f_path + "dogs/dog*.jpg")
length = len(filelist) + 1
label = []
data_label = []

for i in range(1,129):
    data_label.append('pixel' + str(i))

for i in range(1,length):
    img = cv2.imread(os.path.join(f_path,img_name) + str(10 + i) + ".jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128,128))
    cv2.imwrite(os.path.join(f_path,img_name) + "g" + str(10 + i) + ".jpg",gray)
    if(1 <= i <= 10):
        label.append(1)
    elif(11 <= i <= 20):
        label.append(2)

filelist = glob.glob(f_path + "/*g.jpg")
label = np.array(label)
datan = np.array([preprocessing.MinMaxScaler().fit_transform(np.array(Image.open(fname))) for fname in filelist])

batch_size = 4
ntrain = int(datan.shape[0]*0.8)
train_iter = mx.io.NDArrayIter(datan[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(datan[ntrain:, :], label[ntrain:], batch_size)

# Set up the symbolic model
#-------------------------------------------------------------------------------

data = mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 = mx.symbol.Convolution(data = data, kernel = (5, 5), num_filter = 20)
tanh_1 = mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 = mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = (2, 2), stride = (2, 2))
# 2nd convolutional layer
conv_2 = mx.symbol.Convolution(data = pool_1, kernel = (5, 5), num_filter = 50)
tanh_2 = mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 = mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = (2, 2), stride = (2, 2))
# 1st fully connected layer
flatten = mx.symbol.Flatten(data = pool_2)
fc_1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 = mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 = mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
NN_model = mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Device used. CPU in my case.
devices = mx.cpu()

# Training
#-------------------------------------------------------------------------------
train_iter.reset()

# Train the model
model = mx.mod.Module(NN_model,
                        data_names = ['data'],
                        label_names = ['softmax_label'],
                        context = devices)

model.fit(train_iter, eval_data=val_iter, num_epoch = 160,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
            eval_metric = mx.metric.Accuracy(),
            epoch_end_callback = mx.callback.log_train_metric(100))