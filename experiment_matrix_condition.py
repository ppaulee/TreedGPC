import numpy as np
import os
import h5py
import math
import cupy as cp



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def func(a):
    a_gpu = cp.asarray(a)
    if cp.linalg.matrix_rank(cp.asarray(a_gpu)).get() == a.shape[0]:
        # calculate condition of matrix ||A|| * ||A^-1||
        return (True, cp.dot(cp.linalg.norm(a_gpu), cp.linalg.norm(cp.linalg.inv(a_gpu))))
    else:
        return (False, -1)



mnist_base_dir = '/mnt/d/_uni/_thesis/code/treed_gp/_tgpc/experiments/mnist/'
arr = os.listdir(mnist_base_dir)
count_invertible = 0
count = 0
cond_numbers = []
for file in arr:
    f = h5py.File(mnist_base_dir + file, 'r')
    for i in range(8):
        matrix = f[f'kxx_{i}']  
        is_inv = func(matrix[0])   
        count_invertible  = count_invertible + is_inv[0]
        if is_inv[0]:
            cond_numbers.append(float(is_inv[1]))
        count = count + 1
cond_numbers = np.array(cond_numbers)
print(f"{count_invertible}/{count} matrices were invertible")
print(cond_numbers)
print(np.mean(cond_numbers))

print('###############################################')

micro_base_dir = '/mnt/d/_uni/_thesis/code/treed_gp/_tgpc/experiments/micro/'
arr = os.listdir(micro_base_dir)
count_invertible = 0
count = 0
cond_numbers = []
for file in arr:
    f = h5py.File(micro_base_dir + file, 'r')
    for i in range(8):
        matrix = f[f'kxx_{i}']    
        is_inv = func(matrix[0])   
        count_invertible  = count_invertible + is_inv[0]
        if is_inv[0]:
            cond_numbers.append(float(is_inv[1]))
        count = count + 1
cond_numbers = np.array(cond_numbers)
print(f"{count_invertible}/{count} matrices were invertible")
print(cond_numbers)
print(np.mean(cond_numbers))


    
