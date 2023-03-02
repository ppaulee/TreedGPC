from cnn_gp import Sequential, Conv2d, ReLU
import torch
import scipy
from sklearn import tree
import numpy as np
import h5py
import uuid
from patchify import patchify
from typing import Tuple
import pickle
from sklearn.metrics import classification_report
from utils import revert_one_hot_encoding
from pathlib import Path
import matplotlib.image
from scipy.special import expit
from sklearn.decomposition import PCA
import cv2
from sklearn.metrics import jaccard_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


try:
    import cupy as cp
except ImportError:
    print("no cupy found")

class TreedGaussianProcessClassifier:

    dir_kernel_matrix = "/kernel_matrix/"
    #dir_kernel_matrix = '/experiments/micro/'
    dir_deicison_tree = "/decision_tree/"
    dir_main = "./_tgpc/"

    # defines the threshold to treat an images as only background
    non_zero_ratio = 0.9

    prediction_count = 0

    def __init__(self, num_classes : int = 6, kernel = None, max_depth : int = 5, filename_kxx : str = None, 
        filename_kzx : str = None, filename_tree : str = None, cuda : bool = False, use_PCA = False,
        var_bias = None, var_weight = None, kernel_size_f = None, kernel_size_l = None, kernel_size = None,
        num_layer = None, verbose = 0):
        """
        __init__ initializes a new instance of a TreedGaussianProcessClassifier

        :param kernel: convolutional kernel from the cnn_gp package. You can also use the create_kernel function to create one
        :param num_classes: number of different classes
        :param max_depth: max depth of the decision tree. If max_depth == 0 then it behaves like a normal Gaussian process classifier
        :param filename_kxx: name of the file of the kxx kernel matrix. If provided the kxx kernel matrix will not be calculated
        :param filename_kzx: name of the file of the kzx kernel matrix. If provided the kzx kernel matrix will not be calculated
        :param cuda: indicates whether to use the gpu for kernel calculations
        :param verbose: 2->everything, 1->restricted, 0->nothing

        optional kernel parameter if kernel is not given
        :param var_bias: var_bias
        :param var_weight: var_weight
        :param kernel_size_f: kernel size of first layer
        :param kernel_size_l: kernel size of last layer
        :param kernel_size: kernel size
        :param num_layer: number of layers
        """
        # sanity checks
        if num_classes <= 0:
            raise ValueError(f"num_classes must be greater than 0")
        if max_depth < 0:
            raise ValueError(f"max_depth can not be negative")
        if not isinstance(filename_kxx, str) and filename_kxx is not None:
            raise ValueError(f"filename_kxx must be a string")
        if not isinstance(filename_kzx, str) and filename_kzx is not None:
            raise ValueError(f"filename_kzx must be a string")
        if not isinstance(filename_tree, str) and filename_tree is not None:
            raise ValueError(f"filename_tree must be a string")
        if cuda:
            assert torch.cuda.is_available() == True            

        self.use_PCA = use_PCA
        self.verbose = verbose
        
        if kernel is None:
            self.kernel = self.create_kernel(kernel_size_f, kernel_size_l, kernel_size, num_layer, var_bias, var_weight)
        else:
            self.kernel = kernel

        if cuda:
            self.kernel = self.kernel.cuda()
        self.cuda = cuda

        self.num_classes = num_classes
        self.max_depth = max_depth

        self.filename_kxx = f"kxx_{uuid.uuid4().hex}"
        self.filename_kzx = f"kzx_{uuid.uuid4().hex}"

        self.kxx_exists = False
        self.kzx_exists = False
        self.tree_exists = False

        if filename_kxx is not None:
            self.filename_kxx = filename_kxx
            self.kxx_exists = True
        else:
            f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kxx, 'w')
        if filename_kzx is not None:
            self.filename_kzx = filename_kzx
            self.kzx_exists = True
        else:
            f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kzx, 'w')
        if filename_tree is not None:
            self.filename_tree = filename_tree
            self.tree_exists = True
    
    def __del__(self):
        # delete h5py file
        pass
    
    def fit(self, train_x : np.ndarray, train_y : np.ndarray, patch_size : Tuple[int, int] = (60,60), 
        stride : int = 60, batch_size : int = 50) -> Tuple[int, int]:
        """
        fit fits a treed Gaussian process classifier to the given data

        :param train_x: Data to train the classifier (num_samples, width, height)
        :param train_y: Groundtruth of train_x (one-hot-encoded) (num_samples, width, height, num_classes)
        :param patch_size: window size of the patch. If (x,y) == (x_width,x_height) then no patches will be created
        :param stride: step to move the window of the patches
        :param batch_size: size of batch that is used to calculate the kernel matrix

        :return (x_padding, y_padding)
        """ 
        # sanity checks
        shape_x = train_x.shape
        shape_y = train_y.shape
        
        if shape_x[0] != shape_y[0]:
            raise ValueError(f"train_x and train_y the same number of samples - {shape_x[0]} is not {shape_y[0]}")
        if shape_x[1] != shape_y[1]:
            raise ValueError(f"train_x and train_y must be same dimensions - {shape_x[1]} is not {shape_y[1]}")
        if shape_x[2] != shape_y[2]:
            raise ValueError(f"train_x and train_y must be same dimensions - {shape_x[2]} is not {shape_y[2]}")
        if patch_size[0] < 0 or patch_size[1] < 0:
            raise ValueError(f"patch_size must be non negative")
        if stride < 0:
            raise ValueError(f"stride must be non negative")
        if batch_size < 0:
            raise ValueError(f"batch_size must be non negative")     

        if (self.max_depth == 0):
            self.tree = tree.DecisionTreeClassifier(min_samples_split=100000000000)
        else:
            self.tree = tree.DecisionTreeClassifier(max_depth=self.max_depth)

        # create patches            
        self.train_x = []
        self.train_y = []
        self.patch_size = patch_size
        self.stride = stride

        # add padding    
        x_padding = patch_size[0] * (train_x.shape[1] // patch_size[0] + 1) -  train_x.shape[1]
        y_padding = patch_size[1] * (train_x.shape[2] // patch_size[1] + 1) - train_x.shape[2]
        self.original_x_shape = train_x.shape
        self.original_y_shape = train_y.shape
        if train_x.shape[1] % patch_size[0] == 0:
            x_padding = 0
        if train_x.shape[2] % patch_size[1] == 0:
            y_padding = 0

        if self.verbose > 0:
            print(f"padding: {(x_padding, y_padding)}")

        train_x_padded = np.pad(train_x, ((0, 0),(0, x_padding),(0, y_padding), (0, 0)), 'constant')
        train_y_padded = np.pad(train_y, ((0, 0),(0, x_padding),(0, y_padding), (0, 0)), 'constant')

        for image in train_x_padded:
            tmp = patchify(image, patch_size, stride)
            self.train_x.append(np.array(tmp).reshape(-1, patch_size[0], patch_size[1], patch_size[2]))
        for image in train_y_padded:
            #print(train_y.shape)
            tmp = patchify(image, (patch_size[0], patch_size[1], self.num_classes), stride)
            self.train_y.append(np.array(tmp).reshape(-1, patch_size[0], patch_size[1], self.num_classes))
        self.train_y = np.array(self.train_y).reshape(-1, patch_size[0], patch_size[1], self.num_classes)
        self.train_x = np.array(self.train_x).reshape(-1, patch_size[0], patch_size[1], patch_size[2])

        # subsamples images, i.e., remove mainly background images
        subsampled_indices = self.__subsample_images(self.train_x)  
        self.train_y = self.train_y[subsampled_indices]
        self.train_x = self.train_x[subsampled_indices]
        if self.verbose > 1:
            print("created image patches")

        # train decision tree
        shape_x = self.train_x.shape
        shape_y = self.train_y.shape
        train_x_vec = self.train_x.reshape(shape_x[0], shape_x[1] * shape_x[2] * shape_x[3])
        train_y_vec_ohe = self.train_y.reshape(shape_y[0], shape_y[1] * shape_y[2], shape_y[3])

        # revert one hot encoding
        train_y_vec = np.zeros((shape_y[0], shape_y[1] * shape_y[2]))
        for i in range(len(train_y_vec_ohe)):
            train_y_vec[i] = list(map(lambda a: np.argmax(a), train_y_vec_ohe[i]))       

        # if file for decision tree is provided load it
        if self.tree_exists is False:
            if self.verbose > 1:
                print("start to fit tree")
            if self.use_PCA:
                self.pca = PCA(n_components=3)
                train_x_vec = self.pca.fit_transform(train_x_vec)
            
            self.tree.fit(train_x_vec, train_y_vec)
            # save decision tree to disk
            with open(self.dir_main + self.dir_deicison_tree + f'model_{uuid.uuid4().hex}.pkl','wb') as f:
                pickle.dump(self.tree, f)
            if self.verbose > 1:
                print("tree fitted")
        else: 
            if self.verbose > 1:
                print("load decision tree from disk")
            # load from disk
            with open(self.dir_main + self.dir_deicison_tree + self.filename_tree, 'rb') as f:
                self.tree = pickle.load(f)
            if self.verbose > 0:
                print("successfully loaded decision tree from disk")

        del train_y_vec
        del train_y_vec_ohe

        # each image corresponds to a leaf node in the decision tree
        # prediction_node_id defines the node_id to the corresponding image
        # node_id ranges from 0 to the number of nodes in the tree
        prediction_node_id = self.tree.apply(train_x_vec)
        # maps [0,num_leaf_ids] -> node_id
        self.leaf_id_to_node_id = np.unique(prediction_node_id)

        self.node_id_to_leaf_id = np.full(self.leaf_id_to_node_id.max()+1, -1, dtype=int)
        for i in range(len(self.leaf_id_to_node_id)):
            node_id = self.leaf_id_to_node_id[i]
            self.node_id_to_leaf_id[node_id] = i

        # maps image_id to the corresponding leaf id
        self.image_id_to_leaf_id = np.zeros(len(train_x_vec), dtype=np.int64)
        for i in range(0, len(train_x_vec)):
            # node_id is the id from the decision tree
            node_id = prediction_node_id[i]
            # leaf_id starts from 1
            leaf_id = np.where(self.leaf_id_to_node_id == node_id)[0][0]
            self.image_id_to_leaf_id[i] = leaf_id

        # create a list index for the image
        # buckets[leaf_id] -> image_id
        self.buckets = [[] for _ in range(len(self.leaf_id_to_node_id))]
        for i in range(len(self.image_id_to_leaf_id)):
            leaf_id = self.image_id_to_leaf_id[i]
            self.buckets[leaf_id].append(i)  
        
        # train_x has shape [N_images, N_channels, img_width, img_height]
        if self.kxx_exists is False:
            for idx, bucket in enumerate(self.buckets):
                train_x_bucket = self.train_x[bucket]
                self.__compute_kernel_matrix(train_x_bucket, batch_size = batch_size, leaf_id=idx, calc_c = True)
        
        del train_x_vec

        return (x_padding, y_padding)

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        predict predicts a new data instance

        param X: new data instance; array of shape (x,y)

        return: segmented image
        """
        #if self.train_x.shape[1:] != X.shape[1:]:
        #    raise ValueError(f"Shape of X must be the same as for the training data")
        x_padding = self.patch_size[0] * (X.shape[1] // self.patch_size[0] + 1) -  X.shape[1]
        y_padding = self.patch_size[1] * (X.shape[2] // self.patch_size[1] + 1) - X.shape[2]
        if X.shape[1] % self.patch_size[0] == 0:
            x_padding = 0
        if X.shape[2] % self.patch_size[1] == 0:
            y_padding = 0

        #train_x_padded = train_x_padded + train_x
        X = np.pad(X, ((0, 0),(0, x_padding),(0, y_padding), (0, 0)), 'constant')

        patches = patchify(X[0], self.patch_size, step=self.stride)
        shape = patches.shape
     

        # patches[y][x]
        result = np.zeros((X.shape[1], X.shape[2], self.num_classes))
        for i in range(shape[0]):
            for j in range(shape[1]):
                
                predicted = self.__predict_raw(np.array(patches[i][j]), proba=True)
                x_pad = (self.stride * i, X.shape[1] - self.patch_size[0] - self.stride * i)
                y_pad = (self.stride * j, X.shape[2] - self.patch_size[1] - self.stride * j)

                patch = np.pad(predicted, (x_pad, y_pad, (0,0)), 'constant', constant_values=0)
                result = np.add(result, patch)

        return np.argmax(result, axis=2)

    def __predict_raw(self, X : np.ndarray, proba = False) -> np.ndarray:
        """
        __predict_raw predicts a new data instance

        param X: new data instance; array of shape (n,x,y)
        param proba: return raw probabilities instead of class labels

        return: segmented image of size (x,y) if proba True then size is (x,y,num_classes)
        """
        # compute kzx for the corresponding leaf node
        X_vec = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        if self.use_PCA:
            X_vec = self.pca.transform(X_vec)
        node_id_prediction = self.tree.apply(X_vec)[0]
        #leaf_id_1 = np.where(self.leaf_id_to_node_id == node_id_prediction)[0][0]
        leaf_id = self.node_id_to_leaf_id[node_id_prediction]
        #print(leaf_id_1 == leaf_id)
        X_bucket = self.buckets[leaf_id]

        #exit()
        if self.kzx_exists is False:                      
            self.__compute_kernel_matrix(X, Z=self.train_x[X_bucket], leaf_id="pred", batch_size = 100)                                                                 

        tmp = self.train_y.reshape(self.train_y.shape[0], self.train_y.shape[1] * self.train_y.shape[2], self.num_classes)
        tmp = tmp[X_bucket]

        f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kxx, 'r')
        dset_kxx = f[f'kxx_c_{leaf_id}']
        f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kzx, 'r')
        dset_kzx = f[f'kzx_pred_{self.prediction_count-1}']
        kzx = dset_kzx[0:dset_kzx.shape[0],0:dset_kzx.shape[1]]

        #print(f"Predict {self.prediction_count-1}")
        # shape y: (width, height, num_classes) one-hot-encoded
        # f = Kzx @ Kxx^-1 @ y 
        # Kxx @ c = y => c = Kxx^-1 @ y
        # => f = Kzx @ c
        c = dset_kxx[0]

        #if self.cuda:
        if self.cuda:
            try:
                res = cp.dot(kzx, c)
            except:
                if self.verbose > 0:
                    print("cuda not available")
        else: 
            res = np.dot(kzx, c)
        # reshape for a one-hot-encoding
        res = res.reshape(self.train_y.shape[1] * self.train_y.shape[2], self.num_classes)

        if proba:
            res = self.__sigmoid(res)
            return res.reshape(X.shape[1], X.shape[2], self.num_classes)

        # perform arg max to find predicted class
        result = np.array(list(map(np.argmax, res)))
        del res
        return result.reshape(X.shape[1], X.shape[2])

    def eval_performance(self, test_x, groundtruth) -> None:
        """
        eval_performance evaluates the performance of the model using classification_report from scikit-learn

        :param test_x: array of test images (num_images, height, width)
        :param groundtruth: groundtruth of test_x (one-hot-encoded) (num_images, height, width, num_classes)
        """
        prediction = np.zeros((len(test_x), test_x.shape[1], test_x.shape[2]))
        for idx, val in enumerate(test_x):
            if self.verbose > 0:
                print(f"Predict {idx+1}/{len(test_x)}")
            v = val.reshape(1, test_x.shape[1], test_x.shape[2], test_x.shape[3])
            prediction[idx] = self.predict(v)
        prediction = prediction.reshape(len(test_x) * test_x.shape[1] * test_x.shape[2] * test_x.shape[3])
        groundtruth = revert_one_hot_encoding(groundtruth)
        groundtruth = groundtruth.reshape(len(groundtruth) * groundtruth.shape[1] * groundtruth.shape[2])

        target_names = []
        target_names += ['Background']
        for i in range(self.num_classes-1):
            target_names += [f"Class {i}"]

        print(jaccard_score(groundtruth, prediction, average = None))
        return classification_report(groundtruth, prediction, output_dict = True)

    def __relu(self, arr : np.ndarray) -> np.ndarray:
        """
        __relu performs a relu function f(x)=max(0,x)

        :param arr: 1-dimensional double array

        :return: f(x)=max(0,x)
        """
        return np.maximum(arr, np.zeros(len(arr)))

    def __sigmoid(self, arr : np.ndarray) -> np.ndarray:
        return expit(arr)
 
    def __divide_in_classes(self, arr : np.ndarray, index : int) -> np.ndarray:
        """
        __divide_in_classes divides a given list into classes 
            (num_samples, num_features, num_classes) -> (num_samples, num_features) at given index
            This is used for a one-vs-rest classifier setting.

        example:
        index = 0: [ [ [1,0,1], [0,0.567,1] ] ] ---> [ [ 1, 0 ] ]
        index = 1: [ [ [1,0,1], [0,0.567,1] ] ] ----> [ [ 0, 1 ] ]
        index = 3: [ [ [1,0,1], [0,0.567,1] ] ] ----> [ [ 1, 1 ] ]

        :param arr: List to divide
        :param index: class index

        :return: new list
        """
        result = np.zeros((len(arr), len(arr[0])))
        result[:,:] = np.ceil(arr[:,:,index])
        return result       

    def create_kernel(self, kernel_size_f : int, kernel_size_l : int,
        kernel_size : int, number_layers : int, var_bias : int, var_weight : int):
        """
        create_kernel creates a convolutional kernel for the Gaussian process

        :param kernel_size: kernel size of the Conv2d layer
        :param number_layers: number of Conv2d layers
        :param last_kernel_size: kernel size for the last layer (dense layer)
        :return: convolutional kernel for a Gaussian process
        """ 

        layers = []
        for _ in range(number_layers):  # n_layers
            layers += [
                Conv2d(kernel_size=kernel_size, padding="same", var_weight=var_weight,
                    var_bias=var_bias),
                ReLU(),
            ]
        
        initial_model = Sequential(
            Conv2d(kernel_size=kernel_size_f, padding="same", var_weight=var_weight,
                var_bias=var_bias),
            ReLU(),
            *layers,
            ReLU(),
            Conv2d(kernel_size=kernel_size_l, padding=0, var_weight=var_weight,
                var_bias=var_bias),
        )

        return initial_model

    def __compute_kernel_matrix(self, X : np.ndarray, Z : np.ndarray =None, batch_size : int = 50, 
        leaf_id : int = None, calc_c : bool = False) -> None:
        """
        __compute_kernel_matrix computes the kernel matrix in batches. Because of the fact that
            k(x_1,x_2) = k(x_2,x_1) the kernel matrix is symmetric. Therefore, we can just compute 
            the upper triangular matrix and copy the elements. The matrix is saved in a h5py file.
            The name of this file is "kxx_<uuid>". The filename is stored in self.filename_kxx / 
            self.filename_kzx depending on the matrix

        :param X: matrix X, numpy array of shape (num_samples, width, height)
        :param Z: matrix X, numpy array of shape (num_samples, width, height). If Z is not given it will 
                compute the kernel matrix K(X,X)
        :param batch_size: Number of samples used in one batch to compute the matrix
        :param leaf_id: Leaf ID of the decision tree. Used to name the matrix
        :param calc_c: calculates c for prediction (for kxx)
        """

        same = False
        if Z is None:
            Z = X
            same = True

        if same:
            f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kxx,'r+')
            dset = f.create_dataset(f"kxx_{leaf_id}", (self.num_classes, len(X),len(Z)), dtype='float32')
        else:
            f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kzx,'r+')
            dset = f.create_dataset(f"kzx_{leaf_id}_{self.prediction_count}", (len(X),len(Z)), dtype='float32')
            self.prediction_count = self.prediction_count + 1

        if calc_c:
            # calculates c for prediction for each class
            f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kxx,'r+')
            dset_c = f.create_dataset(f"kxx_c_{leaf_id}", (1, len(X), X.shape[1] * X.shape[2] * self.num_classes), dtype='float32')
            #dset_c = f.create_dataset(f"kxx_c_{leaf_id}", (len(X), X.shape[1] * X.shape[2] * self.num_classes), dtype='float32')

        shape_x = X.shape
        if len(X) == 1:
            for j in range(0, len(Z)+1, batch_size):
                #print(f"Batch: (0,{j}) for leaf_id: {leaf_id}")
                start_j = max(0, j-batch_size)
                end_j = min(j + batch_size, len(Z))
                #print('leaf')
                if self.cuda:
                    #print(X.shape)
                    #print(Z.shape)
                    tensor_x = torch.tensor(np.moveaxis(X, 3, 1), dtype=torch.float32).cuda()
                    tensor_z = torch.tensor(np.moveaxis(Z[start_j:end_j], 3, 1), dtype=torch.float32).cuda()
                    dset[0, start_j:end_j] = self.kernel(tensor_x, tensor_z).cpu()
                else:
                    tensor_x = torch.tensor(np.moveaxis(X, 3, 1), dtype=torch.float32)
                    tensor_z = torch.tensor(np.moveaxis(Z[start_j:end_j], 3, 1), dtype=torch.float32)
                    dset[0, start_j:end_j] = self.kernel(tensor_x, tensor_z)
        else:
            if self.verbose > 0:
                print(f"Calculate kernel matrix with dimensions {len(X),len(Z)} for leaf_id {leaf_id}")
            for i in range(0, len(X)+1, batch_size):
                # use i as a start => calculate only upper triangular matrix
                # kernel matrix is symmetric
                for j in range(i, len(Z)+1, batch_size):
                    if self.verbose > 0:
                        print(f"Batch: {(i,j)} for leaf_id: {leaf_id}")
                    start_i = i
                    end_i = min(i + batch_size, len(X))
                    start_j = j
                    end_j = min(j + batch_size, len(Z))
                    #print(f"start_i: {start_i} - end_i {end_i}")
                    #print(f"start_j: {start_j} - end_j {end_j}")
                    if self.cuda:
                        try:
                            tensor_x = torch.tensor(np.moveaxis(X[start_i:end_i], 3, 1), dtype=torch.float32).cuda()
                            tensor_z = torch.tensor(np.moveaxis(Z[start_j:end_j], 3, 1), dtype=torch.float32).cuda()
                            dset[0, start_i:end_i, start_j:end_j] = self.kernel(tensor_x, tensor_z).cpu()
                            dset[0, start_j:end_j, start_i:end_i] = dset[0, start_i:end_i, start_j:end_j].T
                            del tensor_x
                            del tensor_z
                            torch.cuda.empty_cache()
                        except:
                            if self.verbose > 0:
                                print("cuda not avaiable")
                    else:
                        tensor_x = torch.tensor(np.moveaxis(X[start_i:end_i], 3, 1), dtype=torch.float32)
                        tensor_z = torch.tensor(np.moveaxis(Z[start_j:end_j], 3, 1), dtype=torch.float32)
                        dset[0, start_i:end_i, start_j:end_j] = self.kernel(tensor_x, tensor_z)
                        dset[0, start_j:end_j, start_i:end_i] = dset[0, start_i:end_i, start_j:end_j].T

        #torch.cuda.empty_cache()
        if calc_c:
            if self.verbose > 0:
                print('calculate c')
                print(dset[0].shape)
            # compute kzx for the corresponding leaf node
            X_bucket = self.buckets[leaf_id]                                                         
            tmp = self.train_y.reshape(self.train_y.shape[0], self.train_y.shape[1] * self.train_y.shape[2], self.num_classes)
            tmp = tmp[X_bucket]
            #if self.cuda:
            if self.cuda:
                try:
                    cp_A = cp.asarray(dset[0])
                    cp_b = cp.asarray(tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2]))         
                    dset_c[0] = cp.linalg.lstsq(cp_A, cp_b, rcond=1e-6)[0].get().reshape(dset[0].shape[0], 
                        self.train_y.shape[1] * self.train_y.shape[2] * self.num_classes)
                    del cp_A
                    del cp_b
                except:
                    if self.verbose > 0:
                        print("cuda not available")

            else:
                #dset_c[i] = scipy.linalg.lstsq(dset[0], self.__divide_in_classes(tmp, i), cond=1e-6, check_finite = False)[0]
                cp_A = dset[0]
                cp_b = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2])
                dset_c[0] = scipy.linalg.lstsq(cp_A, cp_b, cond=1e-6, overwrite_a = True, overwrite_b = True, check_finite = False)[0].reshape(dset[0].shape[0], 
                    self.train_y.shape[1] * self.train_y.shape[2] * self.num_classes)
            if self.verbose > 1:    
                print('finished calculating c')

        # clear all GPU memory
        #mempool = cp.get_default_memory_pool()
        #pinned_mempool = cp.get_default_pinned_memory_pool()
        #mempool.free_all_blocks()
        #pinned_mempool.free_all_blocks()

    def display_buckets(self, dir = "./buckets/", prob = 0.1):
        """
        display_buckets saves the images of all decision tree leaves in "<dir>/<leaf_id>"

        :param dir: base directory
        :param prob: probability that the image will be saved
        """
        for idx, bucket in enumerate(self.buckets):
            Path(dir + str(idx)).mkdir(parents=True, exist_ok=True)
            train_x_bucket = self.train_x[bucket]
            n = len(train_x_bucket)
            probs = np.random.binomial(n=1, p=prob, size=n)
            zipped = np.array(list(zip(train_x_bucket,probs)))
            filter = np.asarray([1])
            filtered = zipped[np.in1d(zipped[:, 1], filter)]
            for i, image in enumerate(filtered):
                img = image[0]                
                matplotlib.image.imsave(f'{dir}/{idx}/groundtruth_{i}.png', img.reshape(img.shape[0], img.shape[1]), cmap='gray')    

    def __subsample_images(self, X):
        """
        __subsample_images subsamples the images (and removes background images except of one)

        :param X: images

        :return indices of subsampled images
        """
        if X.shape[3] > 1:
            return np.array(range(len(X)))
        non_zero = np.array(list(map(cv2.countNonZero, X)))
        non_zero = non_zero / (X[0].shape[0] * X[0].shape[1])
        images = list(zip(range(len(X)), non_zero))
        non_background = list(filter(lambda x: x[1] < self.non_zero_ratio, images))
        non_background = np.array(list(map(lambda x: x[0], non_background)))

        background = list(filter(lambda x: x[1] >= self.non_zero_ratio, images))
        background = np.array(list(map(lambda x: x[0], background)))
        if self.verbose > 0:
            print(f"removed {len(background)} background images")

        if (len(background) == 0):
            return non_background

        non_background = np.concatenate((non_background, [background[0]]), axis=0)    
        return non_background               

    def get_filenames(self) -> Tuple[str, str, str]:
        """
        get_filenames returns the filenames

        :return (filename_kxx, filename_kzx, filename_tree)
        """
        return (self.filename_kxx, self.filename_kzx, self.filename_tree)

    def get_path(self) -> str:
        """
        get_path returns the path to the kxx and kzx files

        :return path
        """
        return self.dir_main + self.dir_kernel_matrix

    def set_path_matrix(self, path : str) -> None:
        """
        set_path_matrix sets the path for the matrix files

        :param path: new path
        """
        self.dir_kernel_matrix = path

    def set_path_tree(self, path : str) -> None:
        """
        set_path_tree sets the path for the decision tree files

        :param path: new path
        """
        self.dir_deicison_tree = path

    def set_path_base(self, path : str) -> None:
        """
        set_path_base sets the base directory

        :param path: new path
        """
        self.dir_main = path