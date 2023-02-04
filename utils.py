import numpy as np

def revert_one_hot_encoding(array):
    num, width, length, x = array.shape
    new_train_y = np.zeros((num,width,length))

    for num_images in range(len(array)):
        for i in range(len(array[num_images])):
            for j in range(len(array[num_images][i])):
                new_train_y[num_images][i][j] = np.argmax(array[num_images][i][j])   
    return new_train_y

mapping_class_to_rgb = [
    (255,255,255), #white -> background
    (0,0,255), #blue -> 0
    (255,255,0), #yellow -> 1
    (0,153,51), #green -> 2
    (204,0,204), #purple -> 3
    (255, 0, 0), #brown -> 4
    (0, 0, 0) #black -> 5
]

def class_to_rgb(image):
    reshaped = image.reshape(image.shape[0] * image.shape[1])   
    result = np.zeros((image.shape[0] * image.shape[1],3), dtype=object)
    for i in range(0, len(reshaped)):
        result[i] = mapping_class_to_rgb[int(reshaped[i])]
    return result
    
def add_none_class(image):
    """
    add_none_class adds a new class (in our case background) to the data
        Note that the new class will be class 0

    param image: images of size (num_images, x, y, classes)

    return: images of size (num_images, x, y, classes+1)
    """
    #res = np.pad(image, ((0,0),(0,0),(0,0),(1,0)))
    num, x, y, classes = image.shape
    res = np.zeros((num, x, y, classes+1))
    res[:,:,:,1:] = image
    res = res.reshape((len(image), image.shape[1] * image.shape[2], image.shape[3]+1))
    for i in range(len(image)):
        res[i] = list(map(lambda_map, res[i]))
    return res.reshape(image.shape[0], image.shape[1], image.shape[2], image.shape[3]+1)

def lambda_map(a):
    if all(v == 0 for v in a):
        return [1 if i == 0 else a[i] for i in range(len(a))]
    return a

def parse_mnist(arr):
    tmp = np.zeros((arr.shape[0], arr.shape[1] * arr.shape[2], 2))
    arr_ = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
    for num in range(len(arr)):
        for i in range(len(arr[0])):
            if arr_[num][i] == 0:
                tmp[num][i][0] = 1
                tmp[num][i][1] = 0
            else:
                tmp[num][i][0] = 0
                tmp[num][i][1] = 1
    return tmp.reshape((arr.shape[0], arr.shape[1], arr.shape[2], 2))


