from venv import create
from treed_gp import TreedGaussianProcessClassifier
from cnn_gp import Sequential, Conv2d, ReLU
from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset)
from sklearn.model_selection import KFold
import numpy as np
from utils import (class_to_rgb, add_none_class)


def create_kernel(kernel_size : int, number_layers : int):
    """
    create_kernel creates a convolutional kernel for the Gaussian process

    :param kernel_size: kernel size of the Conv2d layer
    :param number_layers: number of Conv2d layers
    :return: convolutional kernel for a Gaussian process
    """ 

    var_bias = 7.86
    var_weight = 2.79

    layers = []
    for i in range(number_layers):  # n_layers
        layers += [
            Conv2d(kernel_size=kernel_size[i], padding="same", var_weight=var_weight * 7**2,
                var_bias=var_bias),
            ReLU(),
        ]
        
    initial_model = Sequential(
        *layers,
        Conv2d(kernel_size=kernel_size[number_layers], padding=0, var_weight=var_weight,
            var_bias=var_bias),
    )
    return initial_model

num_training_samples = 100
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=1,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=True)



def cross_validate(X, y, kernel, max_depth):
    kf = KFold(n_splits=10)
    f1 = np.zeros(10)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        train_x = X[train_index]
        train_y = y[train_index]
        test_x = X[test_index]
        test_y = y[test_index]

        treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=max_depth, cuda = True)
        tmp = add_none_class(train_y)
        np.ceil(tmp.astype(float), out=tmp)
        # batch size 200 for 8GB of GPU RAM
        treed_gpc.fit(train_x.reshape(len(train_x),60,60), tmp, batch_size = 200, patch_size=(20,20), stride = 10)
        performance = treed_gpc.eval_performance(test_x, test_y)
        f1[i] = performance['macro avg']['f1-score']
    print(np.mean(f1))
    return np.mean(f1)

number_layers = 5
cross = {}
for i in range(5):
    arr = np.repeat(3, 3)
    kernel_arr = np.concatenate(([15], arr, [20]), axis = None)
    print(kernel_arr)
    kernel = create_kernel(kernel_arr, 4)
    cross[str(i)] = cross_validate(train_x, train_y, kernel, i)
print(cross)

'''
kernel layers
{'[15 20]': 0.2184690940640232, 
'[15  3 20]': 0.21767094724214014, 
'[15  3  3 20]': 0.21630948067830102, 
'[15  3  3  3 20]': 0.21487239469901423, 
'[15  3  3  3  3 20]': 0.2134721371416745}

input kernel size
{'10': 0.21722967084063088, 
'11': 0.2165183567807721, 
'12': 0.21647647851863838, 
'13': 0.21665142997206047, 
'14': 0.21673908702296973, 
'15': 0.21693095155407827, 
'16': 0.21640057544269978, 
'17': 0.21625899673318197, 
'18': 0.21615087091716187, 
'19': 0.21643898250595686}

tree height
{'0': 0.20995855527107343, 
'1': 0.21159509416822758, 
'2': 0.21070076597978277, 
'3': 0.20611343486579253, 
'4': 0.20460346720027256}
'''