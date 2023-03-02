from treed_gp import TreedGaussianProcessClassifier
from cnn_gp import Sequential, Conv2d, ReLU
from extended_mnist.semantic_segmentation import create_semantic_segmentation_dataset
import numpy as np
from utils import (add_none_class, parse_microscopy)
from sklearn.utils import resample
import os



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


num_training_samples = 100
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=1,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=True)


var_bias = 7.86
var_weight = 2.79
kernel_mnist = Sequential(
    Conv2d(kernel_size=15, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),  # Total 7 layers before dense
    # Dense Layer
    Conv2d(kernel_size=20, padding=0, var_weight=var_weight, var_bias=var_bias))

arr = os.listdir('./_tpgc/experiments/mnist')
print(arr)
if False:
    for i in range(10):
        treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel_mnist, max_depth=3, cuda = True, verbose = 0)
        train_y = add_none_class(train_y)
        X_sample, y_sample = resample(train_x, train_y, random_state=0, replace=False, n_samples=50)
        treed_gpc.fit(X_sample.reshape(50,60,60,1), y_sample, patch_size = (20,20,1), stride = 5, batch_size=250)


#exit()

train_x, train_y = parse_microscopy('/mnt/d/_uni/_thesis/code/render_images/output_preproc', num=50)

for i in range(10):
    print(i)
    treed_gpc = TreedGaussianProcessClassifier(num_classes = 4, kernel=kernel_mnist, max_depth=3, cuda = True, verbose = 0)
    train_y = add_none_class(train_y)
    X_sample, y_sample = resample(train_x, train_y, random_state=0, replace=False, n_samples=15)
    treed_gpc.fit(X_sample.reshape(15,736,973,3), y_sample, patch_size = (256,256,3), stride = 100, batch_size=50)