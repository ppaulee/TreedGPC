from cnn_gp import Sequential, Conv2d, ReLU
import torch
import scipy
from sklearn import tree
import numpy as np
import h5py
import uuid
from patchify import patchify, unpatchify
from typing import Tuple
import pickle
from sklearn.metrics import classification_report
from utils import revert_one_hot_encoding

class TreedGaussianProcessClassifier:

    dir_kernel_matrix = "/kernel_matrix/"
    dir_deicison_tree = "/decision_tree/"
    dir_main = "./_tgpc/"

    prediction_count = 0

    def __init__(self, kernel, num_classes : int, max_depth : int = 5, filename_kxx : str = None, 
        filename_kzx : str = None, filename_tree : str = None, cuda : bool = False):
        """
        __init__ initializes a new instance of a TreedGaussianProcessClassifier

        :param kernel: convolutional kernel from the cnn_gp package. You can also use the create_kernel function to create one
        :param num_classes: number of different classes
        :param max_depth: max depth of the decision tree. If max_depth == 0 then it behaves like a normal Gaussian process classifier
        :param filename_kxx: name of the file of the kxx kernel matrix. If provided the kxx kernel matrix will not be calculated
        :param filename_kzx: name of the file of the kzx kernel matrix. If provided the kzx kernel matrix will not be calculated
        :param cuda: indicates whether to use the gpu for kernel calculations
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

        self.max_depth = max_depth
        
        self.kernel = kernel
        if cuda:
            self.kernel = kernel.cuda()
        self.cuda = cuda

        self.num_classes = num_classes
        if (max_depth == 0):
            self.tree = tree.DecisionTreeClassifier(min_samples_split=100000000000)
        else:
            self.tree = tree.DecisionTreeClassifier(max_depth=max_depth)

        self.classifier = np.zeros(num_classes)

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
        patch_step : int = 60, batch_size : int = 50) -> None:
        """
        fit fits a treed Gaussian process classifier to the given data

        :param train_x: Data to train the classifier (num_samples, width, height)
        :param train_y: Groundtruth of train_x (one-hot-encoded) (num_samples, width, height, num_classes)
        :param patch_size: window size of the patch. If (x,y) == (x_width,x_height) then no patches will be created
        :param patch_step: step to move the window of the patches
        :param batch_size: size of batch that is used to calculate the kernel matrix
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
        if patch_step < 0:
            raise ValueError(f"patch_step must be non negative")
        if batch_size < 0:
            raise ValueError(f"batch_size must be non negative")       

        # create patches            
        self.train_x = []
        self.train_y = []
        self.patch_size = patch_size
        self.patch_step = patch_step
        for image in train_x:
            tmp = patchify(image, patch_size, patch_step)
            self.train_x.append(np.array(tmp).reshape(-1, patch_size[0], patch_size[1]))
        for image in train_y:
            tmp = patchify(image, (patch_size[0], patch_size[1], self.num_classes), patch_step)
            self.train_y.append(np.array(tmp).reshape(-1, patch_size[0], patch_size[1], self.num_classes))
        self.train_y = np.array(self.train_y).reshape(-1, patch_size[0], patch_size[1], self.num_classes)
        self.train_x = np.array(self.train_x).reshape(-1, patch_size[0], patch_size[1])
        print("created image patches")

        # train decision tree
        train_x_vec = train_x.reshape(shape_x[0], shape_x[1] * shape_x[2])
        train_y_vec_eoh = train_y.reshape(shape_y[0], shape_y[1] * shape_y[2], shape_y[3])

        # revert one hot encoding
        train_y_vec = np.zeros((shape_y[0], shape_y[1] * shape_y[2]))
        for i in range(len(train_y_vec_eoh)):
            train_y_vec[i] = list(map(lambda a: np.argmax(a), train_y_vec_eoh[i]))       

        # if file for decision tree is provided load it
        if self.tree_exists is False:
            print("start to fit tree")
            self.tree.fit(train_x_vec, train_y_vec)
            # save decision tree to disk
            with open(self.dir_main + self.dir_deicison_tree + f'model_{uuid.uuid4().hex}.pkl','wb') as f:
                pickle.dump(self.tree, f)
            print("tree fitted")
        else: 
            print("load decision tree from disk")
            # load from disk
            with open(self.dir_main + self.dir_deicison_tree + self.filename_tree, 'rb') as f:
                self.tree = pickle.load(f)
            print("successfully loaded decision tree from disk")

        # each image corresponds to a leaf node in the decision tree
        # prediction_node_id defines the node_id to the corresponding image
        # node_id ranges from 0 to the number of nodes in the tree
        prediction_node_id = self.tree.apply(train_x_vec)
        # maps [0,num_leaf_ids] -> node_id
        self.leaf_id_to_node_id = np.unique(prediction_node_id)

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

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        predict predicts a new data instance

        param X: new data instance; array of shape (x,y)

        return: segmented image
        """
        # TODO return raw values with probabilites
        # TODO patchify
        #if self.train_x.shape[1:] != X.shape[1:]:
        #    raise ValueError(f"Shape of X must be the same as for the training data")

        patches = patchify(X[0], self.patch_size, step=self.patch_size[0])
        shape = patches.shape
        patches_predict = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                patches_predict[i][j] = self.__predict_raw(np.array([patches[i][j]]))

        return unpatchify(patches_predict, X[0].shape)

    def __predict_raw(self, X : np.ndarray) -> np.ndarray:
        """
        __predict_raw predicts a new data instance

        param X: new data instance; array of shape (1,x,y)

        return: segmented image of size (x,y)
        """

        # compute kzx for the corresponding leaf node
        X_vec = X.reshape(1, X.shape[1] * X.shape[2])
        node_id_prediction = self.tree.apply(X_vec)[0]
        leaf_id = np.where(self.leaf_id_to_node_id == node_id_prediction)[0][0]
        X_bucket = self.buckets[leaf_id]
        if self.kzx_exists is False:                      
            self.__compute_kernel_matrix(X, Z=self.train_x[X_bucket], leaf_id="pred", batch_size = 100)                                                                 

        tmp = self.train_y.reshape(self.train_y.shape[0], self.train_y.shape[1] * self.train_y.shape[2], self.num_classes)
        tmp = tmp[X_bucket]

        # train n different one_vs_rest classifier
        f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kxx, 'r')
        dset_kxx = f[f'kxx_c_{leaf_id}']
        kxx = dset_kxx[0:dset_kxx.shape[0],0:dset_kxx.shape[1]]
        f = h5py.File(self.dir_main + self.dir_kernel_matrix + self.filename_kzx, 'r')
        dset_kzx = f[f'kzx_pred_{self.prediction_count-1}']
        kzx = dset_kzx[0:dset_kzx.shape[0],0:dset_kzx.shape[1]]
        one_vs_rest = []

        print(f"Predict {self.prediction_count-1}")

        for i in range(0, self.num_classes):
            # TODO pre-calculate c in h5py -> do not calculate c for every prediction
            #c = scipy.linalg.lstsq(kxx, self.__divide_in_classes(tmp, i), cond=1e-6, check_finite = False)[0]
            c = dset_kxx[i]
            res = kzx @ c
            one_vs_rest.append(res)
        one_vs_rest = np.array(one_vs_rest)

        # perform arg max over all one_vs_rest classifier to find predicted class
        result = np.zeros(self.train_y.shape[1] * self.train_y.shape[2])
        for i in range(one_vs_rest.shape[2]):
            classes = np.zeros(one_vs_rest.shape[0])
            for c in range(one_vs_rest.shape[0]):
                classes[c] = one_vs_rest[c][0][i]
            result[i] = np.argmax(self.__relu(classes))

        return result.reshape(X.shape[1], X.shape[2])

    def eval_performance(self, test_x, groundtruth) -> None:
        """
        eval_performance evaluates the performance of the model using classification_report from scikit-learn

        :param test_x: array of test images (num_images, height, width)
        :param groundtruth: groundtruth of test_x (one-hot-encoded) (num_images, height, width, num_classes)
        """
        prediction = np.zeros((len(test_x), test_x.shape[1], test_x.shape[2]))
        for idx, val in enumerate(test_x):
            v = val.reshape(1, test_x.shape[1], test_x.shape[2])
            prediction[idx] = self.predict(v)
        prediction = prediction.reshape(len(test_x) * test_x.shape[1] * test_x.shape[2])
        groundtruth = revert_one_hot_encoding(groundtruth)
        groundtruth = groundtruth.reshape(len(groundtruth) * groundtruth.shape[1] * groundtruth.shape[2])

        target_names = []
        for i in range(self.num_classes):
            target_names += [f"Class {i}"]
        
        print(classification_report(groundtruth, prediction, target_names=target_names))

    def __relu(self, arr : np.ndarray) -> np.ndarray:
        """
        __relu performs a relu function f(x)=max(0,x)

        :param arr: 1-dimensional double array

        :return: f(x)=max(0,x)
        """
        return np.maximum(arr, np.zeros(len(arr)))

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

    def create_kernel(self, kernel_size : int, number_layers : int, last_kernel_size : int):
        """
        create_kernel creates a convolutional kernel for the Gaussian process

        :param kernel_size: kernel size of the Conv2d layer
        :param number_layers: number of Conv2d layers
        :param last_kernel_size: kernel size for the last layer (dense layer)
        :return: convolutional kernel for a Gaussian process
        """ 

        var_bias = 7.86
        var_weight = 2.79

        layers = []
        for _ in range(number_layers):  # n_layers
            layers += [
                Conv2d(kernel_size=kernel_size, padding="same", var_weight=var_weight * 7**2,
                    var_bias=var_bias),
                ReLU(),
            ]
        
        initial_model = Sequential(
            *layers,
            Conv2d(kernel_size=last_kernel_size, padding=0, var_weight=var_weight,
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
            dset_c = f.create_dataset(f"kxx_c_{leaf_id}", (self.num_classes, len(X), X.shape[1] * X.shape[2]), dtype='float32')

        shape_x = X.shape
        print(f"Calculate kernel matrix with dimensions {len(X),len(Z)} for leaf_id {leaf_id}")
        if len(X) == 1:
            for j in range(0, len(Z)+1, batch_size):
                #print(f"Batch: (0,{j}) for leaf_id: {leaf_id}")
                start_j = max(0, j-batch_size)
                end_j = min(j + batch_size, len(Z))

                if self.cuda:
                    tensor_x = torch.tensor(X[0].reshape(1, 1, shape_x[1], shape_x[2]), dtype=torch.float32).cuda()
                    tensor_z = torch.tensor(Z[start_j:end_j].reshape(end_j - start_j, 1, shape_x[1], shape_x[2]), dtype=torch.float32).cuda()
                    dset[0, start_j:end_j] = self.kernel(tensor_x, tensor_z).cpu()
                else:
                    tensor_x = torch.tensor(X[0].reshape(1, 1, shape_x[1], shape_x[2]), dtype=torch.float32)
                    tensor_z = torch.tensor(Z[start_j:end_j].reshape(end_j - start_j, 1, shape_x[1], shape_x[2]), dtype=torch.float32)
                    dset[0, start_j:end_j] = self.kernel(tensor_x, tensor_z)
        else:
            for i in range(0, len(X)+1, batch_size):
                # use i as a start => calculate only upper triangular matrix
                # kernel matrix is symmetric
                for j in range(i, len(Z)+1, batch_size):
                    print(f"Batch: {(i,j)} for leaf_id: {leaf_id}")
                    start_i = i
                    end_i = min(i + batch_size, len(X))
                    start_j = j
                    end_j = min(j + batch_size, len(Z))
                    #print(f"start_i: {start_i} - end_i {end_i}")
                    #print(f"start_j: {start_j} - end_j {end_j}")
                    if self.cuda:
                        tensor_x = torch.tensor(X[start_i:end_i].reshape(end_i - start_i, 1, shape_x[1], shape_x[2]), dtype=torch.float32).cuda()
                        tensor_z = torch.tensor(Z[start_j:end_j].reshape(end_j - start_j, 1, shape_x[1], shape_x[2]), dtype=torch.float32).cuda()
                        dset[0, start_i:end_i, start_j:end_j] = self.kernel(tensor_x, tensor_z).cpu()
                        dset[0, start_j:end_j, start_i:end_i] = dset[0, start_i:end_i, start_j:end_j].T
                    else:
                        tensor_x = torch.tensor(X[start_i:end_i].reshape(end_i - start_i, 1, shape_x[1], shape_x[2]), dtype=torch.float32)
                        tensor_z = torch.tensor(Z[start_j:end_j].reshape(end_j - start_j, 1, shape_x[1], shape_x[2]), dtype=torch.float32)
                        dset[0, start_i:end_i, start_j:end_j] = self.kernel(tensor_x, tensor_z)
                        dset[0, start_j:end_j, start_i:end_i] = dset[0, start_i:end_i, start_j:end_j].T

        if calc_c:
            print('calculate c')
            # compute kzx for the corresponding leaf node
            X_bucket = self.buckets[leaf_id]                                                         
            tmp = self.train_y.reshape(self.train_y.shape[0], self.train_y.shape[1] * self.train_y.shape[2], self.num_classes)
            tmp = tmp[X_bucket]
            for i in range(self.num_classes):
                dset_c[i] = scipy.linalg.lstsq(dset[0], self.__divide_in_classes(tmp, i), cond=1e-6, check_finite = False)[0]
            print('finished calculating c')


    def __display_buckets():
        pass

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



