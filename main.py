from treed_gp import TreedGaussianProcessClassifier
from utils import (revert_one_hot_encoding, class_to_rgb, add_none_class, parse_mnist)

from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset, display_segmented_image,
                                                                       display_grayscale_array, plot_class_masks)
from cnn_gp import Sequential, Conv2d, ReLU
import numpy as np
import h5py
import datetime

np.random.seed(seed=9)
start = datetime.datetime.now()
"""
f = h5py.File('./kernel_matrix/kxx_1672142084_1811712', 'r')
print(list(f.keys()))
dset = f['kxx']
print(dset.shape)
exit()
"""

var_bias = 7.86
var_weight = 2.79
kernel = Sequential(
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),
    Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
    ReLU(),  # Total 7 layers before dense
    # Dense Layer
    Conv2d(kernel_size=60, padding=0, var_weight=var_weight, var_bias=var_bias))

#treed_gpc = TreedGaussianProcessClassifier(num_classes = 2, kernel=kernel, filename_kxx="kxx_mnist")

treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, filename_tree="model_9de1a9b1e21d4de692d4c8a8d8b49657.pkl")

num_training_samples = 10

#"""
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=10,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=True)
#"""

"""
import tensorflow as tf
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_y = parse_mnist(train_x)
"""
#train_y = revert_one_hot_encoding(train_y)
#test_y = revert_one_hot_encoding(test_y)

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

tmp = add_none_class(train_y)
tmp = np.ceil(tmp.astype(float))
print("preparing to fit data")

treed_gpc.fit(train_x.reshape(num_training_samples,60,60), tmp, batch_size = 100)

print("finished fit")
print(test_x[0].shape)
result = treed_gpc.predict(test_x[0].reshape(1,60,60))
print(f"shape result: {result.shape}")

result_rgb = class_to_rgb(result).reshape(60,60,3)
result_rgb = result_rgb.astype(np.uint8)

import matplotlib.image
matplotlib.image.imsave('name.png', result_rgb)
matplotlib.image.imsave('groundtruth.png', test_x[0].reshape(test_x[0].shape[0], test_x[0].shape[1]), cmap='gray')

end = datetime.datetime.now()
print(f"total time: {end-start}")


