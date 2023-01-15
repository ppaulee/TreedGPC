from treed_gp import TreedGaussianProcessClassifier
from utils import (class_to_rgb, add_none_class)
from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset, display_segmented_image,
                                                                       display_grayscale_array, plot_class_masks)
from cnn_gp import Sequential, Conv2d, ReLU
import numpy as np
import datetime

np.random.seed(seed=9)

start = datetime.datetime.now()

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

treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=4, cuda = False)

#treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=4, filename_tree="model_10000.pkl",
#   filename_kxx="kxx_10000")

num_training_samples = 10000

#"""
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=10,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=True)

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

tmp = add_none_class(train_y)
tmp = np.ceil(tmp.astype(float))
print("preparing to fit data")

treed_gpc.fit(train_x.reshape(num_training_samples,60,60), tmp, batch_size = 300)

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
# 0:00:58.919398 on 100 images with cuda
# 0:00:26.375704 on 100 images without cuda
#
# 0:03:45.337014 on 1000 images with cuda
# 0:04:13.744935 on 1000 images without cuda
#treed_gpc.eval_performance(test_x[:1000], test_y[:1000])



