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


#treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, cuda = True, filename_tree='model_b27957ab1237454da64143d4a9738373.pkl', filename_kxx='kxx_7236fc3ccf5e4cd8a1a7e97923588f73', verbose=1)
treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, use_PCA=False, cuda = True, verbose=1)

num_training_samples = 200
num_test_samples = 100

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
print("preparing to fit data")

# batch size 200 for 8GB of GPU RAM
treed_gpc.fit(train_x.reshape(num_training_samples,60,60,1), train_y, batch_size = 250, patch_size=(20,20,1), stride = 5)


print("finished fit")
print(test_x[0].shape)
for i in range(20):
    result = treed_gpc.predict(test_x[i].reshape(1,60,60,1))
    print(f"shape result of {i}: {result.shape}")

    result_rgb = class_to_rgb(result).reshape(60,60,3)
    result_rgb = result_rgb.astype(np.uint8)

    import matplotlib.image
    matplotlib.image.imsave(f'./experiments/extended_mnist_non/prediction_{i}_non_exlcusive.png', result_rgb)
    result_rgb_y = class_to_rgb(np.argmax(test_y[i], axis=2)).reshape(60,60,3).astype(np.uint8)
    matplotlib.image.imsave(f'./experiments/extended_mnist_non/original_{i}_non_exlcusive.png', test_x[i].reshape(test_x[i].shape[0], test_x[i].shape[1]), cmap='gray')
    matplotlib.image.imsave(f'./experiments/extended_mnist_non/groundtruth_{i}_non_exlcusive.png', result_rgb_y)
    result_rgb_train_y = class_to_rgb(np.argmax(train_y[i], axis=2)).reshape(60,60,3).astype(np.uint8)
    matplotlib.image.imsave(f'./experiments/extended_mnist_non/train_y_{i}_non_exlcusive.png', result_rgb_train_y)
exit()
end = datetime.datetime.now()
print(f"total time: {end-start}")

start = datetime.datetime.now()
performance = treed_gpc.eval_performance(test_x[:1000].reshape(num_test_samples,60,60,1), test_y[:1000])
print(performance)
end = datetime.datetime.now()
print(f"total time for prediction: {end-start}")

'''
without pca
[0.94046364 0.45997158 0.49650582 0.37681438 0.3796576  0.36598818]
{
   "0.0":{
      "precision":0.9555244218174836,
      "recall":0.9835166531827249,
      "f1-score":0.9693184878821494,
      "support":323842
   },
   "1.0":{
      "precision":0.7047466976339092,
      "recall":0.5697688064781129,
      "f1-score":0.6301103179753408,
      "support":8521
   },
   "2.0":{
      "precision":0.6869244935543278,
      "recall":0.6417204301075269,
      "f1-score":0.663553480097843,
      "support":4650
   },
   "3.0":{
      "precision":0.6162553057695331,
      "recall":0.4923386083898518,
      "f1-score":0.5473713607484465,
      "support":7962
   },
   "4.0":{
      "precision":0.7144210725791169,
      "recall":0.4475839962008785,
      "f1-score":0.5503649635036496,
      "support":8423
   },
   "5.0":{
      "precision":0.733421052631579,
      "recall":0.42214480460466525,
      "f1-score":0.535858488752163,
      "support":6602
   },
   "accuracy":0.9356111111111111,
   "macro avg":{
      "precision":0.7352155073309916,
      "recall":0.5928455498272934,
      "f1-score":0.6494295164932653,
      "support":360000
   },
   "weighted avg":{
      "precision":0.928901457067023,
      "recall":0.9356111111111111,
      "f1-score":0.9302565668312943,
      "support":360000
   }
}

with pca
[0.94107742 0.46758257 0.48715768 0.37959577 0.39505051 0.37613353]
{
   "0.0":{
      "precision":0.9560238053078753,
      "recall":0.9836586977600188,
      "f1-score":0.9696443926976083,
      "support":323842
   },
   "1.0":{
      "precision":0.7023321554770318,
      "recall":0.5831475178969604,
      "f1-score":0.6372146704283149,
      "support":8521
   },
   "2.0":{
      "precision":0.6997312484730027,
      "recall":0.6159139784946237,
      "f1-score":0.6551526935834383,
      "support":4650
   },
   "3.0":{
      "precision":0.6189579409918393,
      "recall":0.4953529264004019,
      "f1-score":0.5502999860471606,
      "support":7962
   },
   "4.0":{
      "precision":0.7258723088344469,
      "recall":0.46432387510388223,
      "f1-score":0.5663601477083484,
      "support":8423
   },
   "5.0":{
      "precision":0.7397260273972602,
      "recall":0.4335049984853075,
      "f1-score":0.5466526597268646,
      "support":6602
   },
   "accuracy":0.9363888888888889,
   "macro avg":{
      "precision":0.7404405810802427,
      "recall":0.595983665690199,
      "f1-score":0.6542207583652891,
      "support":360000
   },
   "weighted avg":{
      "precision":0.9299022793808293,
      "recall":0.9363888888888889,
      "f1-score":0.9312463514567589,
      "support":360000
   }
}
'''