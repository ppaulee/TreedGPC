from treed_gp import TreedGaussianProcessClassifier
from utils import (class_to_rgb, add_none_class, parse_microscopy)
from cnn_gp import Sequential, Conv2d, ReLU
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

np.random.seed(seed=9)

start = datetime.datetime.now()

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
    Conv2d(kernel_size=256, padding=0, var_weight=var_weight, var_bias=var_bias))

treed_gpc = TreedGaussianProcessClassifier(num_classes = 4, kernel=kernel, max_depth=3, cuda = True, use_PCA=True)


num_training_samples = 4

X,y = parse_microscopy('/mnt/d/_uni/_thesis/code/render_images/output_preproc', num_training_samples+1)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=1, random_state=42)


""" pca = PCA(n_components=3)
pca.fit(train_x.reshape(len(train_x), 736*973*3))
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))
exit() """

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

print("Add background class")

train_y = add_none_class(train_y)
test_y = add_none_class(test_y)



print("Finished adding the background class")
print(train_y.shape)
#tmp = np.ceil(tmp.astype(float))
np.ceil(train_y.astype(float), out=train_y)
print("preparing to fit data")

treed_gpc.fit(train_x.reshape(num_training_samples,736,973,3), train_y, batch_size = 30, patch_size=(256,256,3), stride = 200)

print("finished fit")
print(test_x[0].shape)
for i in range(1):
    result = treed_gpc.predict(test_x[i].reshape(1,736,973,3))
    print(f"shape result: {result.shape}")

    result_rgb = class_to_rgb(result).reshape(736,973,3)
    result_rgb = result_rgb.astype(np.uint8)

    import matplotlib.image
    matplotlib.image.imsave(f'm_prediction_{i}.png', result_rgb)
    #matplotlib.image.imsave(f'm_groundtruth_{i}.png', test_x[i].reshape(test_x[i].shape[0], test_x[i].shape[1]), cmap='gray')

end = datetime.datetime.now()
print(f"total time: {end-start}")
