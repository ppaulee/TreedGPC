from multiprocessing import Pool
from treed_gp import TreedGaussianProcessClassifier
from cnn_gp import Sequential, Conv2d, ReLU
from extended_mnist.semantic_segmentation import create_semantic_segmentation_dataset
from sklearn.model_selection import KFold
import numpy as np
from utils import (class_to_rgb, add_none_class)
import h5py
import uuid
import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(seed=9)

num_training_samples = 20
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=1,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=True)
train_y = add_none_class(train_y)
test_y = add_none_class(test_y)
np.ceil(train_y.astype(float), out=train_y)
np.ceil(test_y.astype(float), out=test_y)

def cross_validate(X, y, kernel=None, max_depth = None, var_bias = None, var_weight = None, kernel_size_f = None, kernel_size_l = None, kernel_size = None,
        num_layer = None):
    k = 2
    kf = KFold(n_splits=k)
    f1 = np.zeros(k)
   
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        train_x = X[train_index]
        train_y = y[train_index]
        test_x = X[test_index]
        test_y = y[test_index]

        treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, max_depth=max_depth, cuda = False,
        var_bias = var_bias, var_weight = var_weight, kernel_size_f = kernel_size_f, kernel_size_l = kernel_size_l, kernel_size = kernel_size,
        num_layer = num_layer, verbose=0)

        # batch size 200 for 8GB of GPU RAM
        treed_gpc.fit(train_x.reshape(len(train_x),60,60,1), train_y, batch_size = 200, patch_size=(20,20,1), stride = 10)
        performance = treed_gpc.eval_performance(test_x.reshape(len(test_x),60,60,1), test_y)
        f1[i] = performance['macro avg']['f1-score']
    print(np.mean(f1))
    return np.mean(f1)

cross = {}
param_grid = {
        'num_classes': [6],
        'max_depth': [2], 
        'var_bias': [7.86],
        'var_weight': [2.79 * (7**2)],
        'kernel_size_f': range(5,30,2),
        'kernel_size_l' : [20],
        'kernel_size' : range(3,20,2),
        'num_layer' : [1,2,3,4,5,6,7,8]
}

dims = np.zeros(len(param_grid.keys()), dtype=int)

for idx, key in enumerate(param_grid.keys()):
    dims[idx] = len(param_grid[key])

var_bias = 7.86
var_weight = 2.79
kernel = Sequential(
    Conv2d(kernel_size=15, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),  # Total 7 layers before dense
    # Dense Layer
    Conv2d(kernel_size=20, padding=0, var_weight=var_weight, var_bias=var_bias))

num_classes = 6
max_depth = 3
var_bias = 7.86
var_weight = 2.79
kernel_size_f = 15
kernel_size_l = 20
kernel_size = 3
num_layer = 2


f = h5py.File(f'./_tgpc/cross_validation_{uuid.uuid4().hex}.hdf5','a')

dset = f.create_dataset(f"max_depth", (100,), dtype='float32')
for idx, max_depth_ in enumerate(param_grid['max_depth']):
  print(idx,"max_depth_")
  dset[idx] = cross_validate(train_x, train_y, max_depth = max_depth_, 
    var_bias = var_bias, var_weight = var_weight, kernel_size_f = kernel_size_f, 
    kernel_size_l = kernel_size_l, kernel_size = kernel_size, num_layer = num_layer)
print(dset[:100])
print(np.argmin(dset))

dset = f.create_dataset(f"kernel_size_f", (100,), dtype='float32')
for idx, kernel_size_f_ in enumerate(param_grid['kernel_size_f']):
  print(idx,"kernel_size_f_")
  dset[idx] = cross_validate(train_x, train_y, max_depth = max_depth, 
    var_bias = var_bias, var_weight = var_weight, kernel_size_f = kernel_size_f_, 
    kernel_size_l = kernel_size_l, kernel_size = kernel_size, num_layer = num_layer)
print(dset)
print(np.argmin(dset))

dset = f.create_dataset(f"kernel_size_l", (100,), dtype='float32')
for idx, kernel_size_l_ in enumerate(param_grid['kernel_size_l']):
  print(idx,"kernel_size_l_")
  dset[idx] = cross_validate(train_x, train_y, max_depth = max_depth, 
    var_bias = var_bias, var_weight = var_weight, kernel_size_f = kernel_size_f, 
    kernel_size_l = kernel_size_l_, kernel_size = kernel_size, num_layer = num_layer)
print(dset)
print(np.argmin(dset))

dset = f.create_dataset(f"kernel_size", (100,), dtype='float32')
for idx, kernel_size_ in enumerate(param_grid['kernel_size']):
  print(idx,"kernel_size_")
  dset[idx] = cross_validate(train_x, train_y, max_depth = max_depth, 
    var_bias = var_bias, var_weight = var_weight, kernel_size_f = kernel_size_f, 
    kernel_size_l = kernel_size_l, kernel_size = kernel_size_, num_layer = num_layer)
print(dset)
print(np.argmin(dset))

dset = f.create_dataset(f"num_layer", (100,), dtype='float32')
for idx, num_layer_ in enumerate(param_grid['num_layer']):
  print(idx,"num_layer_")
  dset[idx] = cross_validate(train_x, train_y, max_depth = max_depth, 
    var_bias = var_bias, var_weight = var_weight, kernel_size_f = kernel_size_f, 
    kernel_size_l = kernel_size_l, kernel_size = kernel_size, num_layer = num_layer_)
print(dset)
print(np.argmin(dset))

                          


'''
Traceback (most recent call last):
  File "/u/halle/ungerman/home_at/bachelor_thesis/code/TreedGPC/cross_validation.py", line 81, in <module>
    dset[id1, id2, id3, id4, id5, id6, id7, id8] = cross_validate(train_x, train_y, max_depth = max_depth,
  File "/u/halle/ungerman/home_at/bachelor_thesis/code/TreedGPC/cross_validation.py", line 42, in cross_validate
    treed_gpc.fit(train_x.reshape(len(train_x),60,60,1), tmp, batch_size = 200, patch_size=(20,20,1), stride = 10)
  File "/u/halle/ungerman/home_at/bachelor_thesis/code/TreedGPC/treed_gp.py", line 259, in fit
    self.__compute_kernel_matrix(train_x_bucket, batch_size = batch_size, leaf_id=idx, calc_c = True)
  File "/u/halle/ungerman/home_at/bachelor_thesis/code/TreedGPC/treed_gp.py", line 564, in __compute_kernel_matrix
    dset_c[0] = scipy.linalg.lstsq(cp_A, cp_b, cond=1e-6, overwrite_a = True, overwrite_b = True, check_finite = False)[0].reshape(dset[0].shape[0],
  File "/usr/lib/python3/dist-packages/scipy/linalg/_basic.py", line 1204, in lstsq
    x, s, rank, info = lapack_func(a1, b1, lwork,
ValueError: On entry to DLASCL parameter number 4 had an illegal value
'''