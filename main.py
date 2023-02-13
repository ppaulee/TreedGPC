from treed_gp import TreedGaussianProcessClassifier
from utils import (class_to_rgb, add_none_class)
from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset, display_segmented_image,
                                                                       display_grayscale_array, plot_class_masks)
from cnn_gp import Sequential, Conv2d, ReLU
import numpy as np
import datetime

# run renders
# python3.10 /mnt/d/_uni/_thesis/code/blender_images/blender/blenderconfig/main.py "/home/paul/blender-3.0.0-linux-x64/blender" "/mnt/d/_uni/_thesis/code/blender_images/blender/cell03_original.blend"
# python D:\\_uni\\_thesis\\code\\blender_images\\blender\\blenderconfig\\main.py "D:\\Program Files\\Blender Foundation\\Blender 3.4\\3.4\\python\\bin\\python.exe" "D:\\_uni\\_thesis\\code\\blender_images\\blender\\cell03_original.blend"


#python D:/_uni/_thesis/code/blender_images/blender/blenderconfig/main.py "D:/Program Files/Blender Foundation/Blender 3.4/3.4/python/bin/python.exe" "D:/_uni/_thesis/code/blender_images/blender/cell03_original.blend"

'''
"D:\\Program Files\\Blender Foundation\\Blender 3.4\\3.4\\python\\bin\\python.exe" 
"D:\\_uni\\_thesis\\code\\blender_images\\blender\\cell03_new.blend"  
--background 
--python "D:\\_uni\\_thesis\\code\\blender_images\\blender\\blenderconfig\\blender_render.py" 
-- "D:\\_uni\\_thesis\\code\\blender_images\\blender\\renders_config\\render_config.00446.json"
'''

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
    Conv2d(kernel_size=20, padding=0, var_weight=var_weight, var_bias=var_bias))

#treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=4, cuda = True, filename_tree='tree_10000.pkl', 
#   filename_kxx='kxx_10000_')

# macro avg f1: 0.19986049243837578
#treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, cuda = True, filename_tree = "model_c8283499b84c4c20a749f6c6885db5b3.pkl", filename_kxx = "kxx_e924d0a0de8040968f4421a16f45fd0d")
treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, cuda = True)

#treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=4, filename_tree="model_10000.pkl",
#   filename_kxx="kxx_10000")

num_training_samples = 200

#"""
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=20,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=True)

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

# batch size 200 for 8GB of GPU RAM
treed_gpc.fit(train_x.reshape(num_training_samples,60,60,1), train_y, batch_size = 500, patch_size=(20,20,1), stride = 5)

#treed_gpc.display_buckets()

print("finished fit")
print(test_x[0].shape)
for i in range(5):
    result = treed_gpc.predict(test_x[i].reshape(1,60,60,1))
    print(f"shape result: {result.shape}")

    result_rgb = class_to_rgb(result).reshape(60,60,3)
    result_rgb = result_rgb.astype(np.uint8)

    import matplotlib.image
    matplotlib.image.imsave(f'prediction_{i}.png', result_rgb)
    matplotlib.image.imsave(f'groundtruth_{i}.png', test_x[i].reshape(test_x[i].shape[0], test_x[i].shape[1]), cmap='gray')

end = datetime.datetime.now()
print(f"total time: {end-start}")
exit()
# 0:00:58.919398 on 100 images with cuda
# 0:00:26.375704 on 100 images without cuda
#
# 0:03:45.337014 on 1000 images with cuda
# 0:04:13.744935 on 1000 images without cuda
# 0:03:24.882039 on 1000 images with cuda and cupy
#
# 0:31:37.725436 on 10000 images with cuda
# 2:01:10.047131 on 10000 images without cuda
# 0:30:52.478438 on 10000 images with cuda and cupy

start = datetime.datetime.now()
performance = treed_gpc.eval_performance(test_x[:1000].reshape(num_training_samples,60,60,1), test_y[:1000])
#performance = treed_gpc.eval_performance(test_x[3], test_y[3])
""" 
matplotlib.image.imsave(f'test___x__{i}.png', test_x[3].reshape(test_x[3].shape[0], test_x[3].shape[1]), cmap='gray')

result_rgb = class_to_rgb(test).reshape(60,60,3)
result_rgb = result_rgb.astype(np.uint8)
matplotlib.image.imsave(f'test__y__{i}.png', result_rgb) 
"""

print(performance)
end = datetime.datetime.now()
print(f"total time for prediction: {end-start}")