from treed_gp import TreedGaussianProcessClassifier
from utils import (class_to_rgb, add_none_class)
from extended_mnist.semantic_segmentation import create_semantic_segmentation_dataset
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

def f(gpu, combination, train_x, train_y, test_x, test_y, kernel):
    len_train = combination[0]
    len_test = combination[1]
    train_x = train_x[:len_train,:,:,:]
    train_y = train_y[:len_train,:,:,:]
    test_x = test_x[:len_test,:,:,:]

    start = datetime.datetime.now()
    treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, cuda = gpu, verbose=0)
    treed_gpc.fit(train_x.reshape(len_train,60,60,1), train_y, batch_size = 250, patch_size=(20,20,1), stride = 5)
    end = datetime.datetime.now()
    print(f"total training time for combination {combination} and GPU is {gpu}: {end-start}")

    start = datetime.datetime.now()
    for i in range(len(test_x)):
        result = treed_gpc.predict(test_x[i].reshape(1,60,60,1))
    end = datetime.datetime.now()
    print(f"total prediction time for combination {combination} and GPU is {gpu}: {end-start}")


 


#treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, cuda = True, filename_tree='model_b27957ab1237454da64143d4a9738373.pkl', filename_kxx='kxx_7236fc3ccf5e4cd8a1a7e97923588f73', verbose=1)

num_training_samples = 100
num_test_samples = 50

#"""
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

# batch size 200 for 8GB of GPU RAM
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
    f(False, combination, train_x, train_y, test_x, test_y, kernel_)
    f(True, combination, train_x, train_y, test_x, test_y, kernel_)
    

