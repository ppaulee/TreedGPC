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
treed_gpc = TreedGaussianProcessClassifier(num_classes = 6, kernel=kernel, max_depth=3, use_PCA=False, cuda = False, verbose=1)

num_training_samples = 8
num_test_samples = 2

#"""
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=num_training_samples,
                                                                        num_test_samples=num_test_samples,
                                                                        image_shape=(60, 60),
                                                                        num_classes=5,
                                                                        labels_are_exclusive=False)
start = datetime.datetime.now()
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
end = datetime.datetime.now()
print(f"total time: {end-start}")
print(test_x[0].shape)
start = datetime.datetime.now()
for i in range(0):
    result = treed_gpc.predict(test_x[i].reshape(1,60,60,1))
    print(f"shape result of {i}: {result.shape}")

    result_rgb = class_to_rgb(result).reshape(60,60,3)
    result_rgb = result_rgb.astype(np.uint8)

    import matplotlib.image
    matplotlib.image.imsave(f'./experiments/extended_mnist_tree/prediction_{i}_non_exlcusive.png', result_rgb)
    result_rgb_y = class_to_rgb(np.argmax(test_y[i], axis=2)).reshape(60,60,3).astype(np.uint8)
    matplotlib.image.imsave(f'./experiments/extended_mnist_tree/original_{i}_non_exlcusive.png', test_x[i].reshape(test_x[i].shape[0], test_x[i].shape[1]), cmap='gray')
    matplotlib.image.imsave(f'./experiments/extended_mnist_tree/groundtruth_{i}_non_exlcusive.png', result_rgb_y)

end = datetime.datetime.now()

print(f"total time prediction: {end-start}")

start = datetime.datetime.now()
performance = treed_gpc.eval_performance(test_x[:1000].reshape(num_test_samples,60,60,1), test_y[:1000])
print(performance)
end = datetime.datetime.now()
print(f"total time for prediction: {end-start}")

'''
without pca without exclusive labels
[0.97701374 0.68018822 0.72286208 0.54877309 0.66078067 0.58468075]
{
   "0.0":{
      "precision":0.9779221304211276,
      "recall":0.999050154389621,
      "f1-score":0.9883732441618727,
      "support":314788
   },
   "1.0":{
      "precision":0.7999236714054002,
      "recall":0.8196304624107928,
      "f1-score":0.8096571704490584,
      "support":10229
   },
   "2.0":{
      "precision":0.867934312878133,
      "recall":0.812196700097056,
      "f1-score":0.8391409710035931,
      "support":6182
   },
   "3.0":{
      "precision":0.8445423005070724,
      "recall":0.6104359567901234,
      "f1-score":0.7086552457731498,
      "support":10368
   },
   "4.0":{
      "precision":0.854380598645401,
      "recall":0.7446443873179092,
      "f1-score":0.795747062115277,
      "support":10503
   },
   "5.0":{
      "precision":0.9012188466436238,
      "recall":0.6247162673392181,
      "f1-score":0.7379161391226633,
      "support":7930
   },
   "accuracy":0.9638833333333333,
   "macro avg":{
      "precision":0.8743203100834597,
      "recall":0.7684456547241201,
      "f1-score":0.8132483054376024,
      "support":360000
   },
   "weighted avg":{
      "precision":0.9618405005587174,
      "recall":0.9638833333333333,
      "f1-score":0.9615398150758628,
      "support":360000
   }
}


'''