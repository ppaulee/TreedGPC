from treed_gp import TreedGaussianProcessClassifier
from utils import (class_to_rgb, add_none_class)
from extended_mnist.semantic_segmentation import create_semantic_segmentation_dataset
from cnn_gp import Sequential, Conv2d, ReLU
import numpy as np
import datetime

np.random.seed(seed=9)

def f(combination, train_x, train_y, test_x, test_y, kernel, use_pca):
    len_train = combination[0]
    len_test = combination[1]
    train_x = train_x[:len_train,:,:,:]
    train_y = train_y[:len_train,:,:,:]
    test_x = test_x[:len_test,:,:,:]
    test_y = test_y[:len_test,:,:,:]

    treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, cuda = True, verbose=0, use_PCA=use_pca)
    treed_gpc.fit(train_x.reshape(len_train,60,60,1), train_y, batch_size = 250, patch_size=(20,20,1), stride = 5)
    print('############################################################################')
    print(f'use PCA is {use_pca} for combination: {combination}')
    performance = treed_gpc.eval_performance(test_x.reshape(len_test,60,60,1), test_y)   
    print(performance)
    print(performance['macro avg']['f1-score'])

num_training_samples = 100
num_test_samples = 50

train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=num_test_samples,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=False)



print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

print("Add background class")
train_y = add_none_class(train_y)
test_y = add_none_class(test_y)


print("Finished adding the background class")
print(train_y.shape)

np.ceil(train_y.astype(float), out=train_y)
print("preparing to fit data")

s = [(8,2),(16,4),(24,6),(32,8),(40,10),(48,12),(56,14), (64,16), (72,18), (80,20)]
for combination in s:
    var_bias = 7.86
    var_weight = 2.79
    kernel_ = Sequential(
        Conv2d(kernel_size=15, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
        ReLU(),
        Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
        ReLU(),
        Conv2d(kernel_size=3, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
        ReLU(),  # Total 7 layers before dense
        # Dense Layer
        Conv2d(kernel_size=20, padding=0, var_weight=var_weight, var_bias=var_bias))
    f(combination, train_x, train_y, test_x, test_y, kernel_, True)
    f(combination, train_x, train_y, test_x, test_y, kernel_, False)
    

