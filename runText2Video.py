import sys
#sys.path.append('/home/liunx_one/Text2Video/natural/tf_mesh_renderer/mesh_renderer')

import numpy as np
import os, os.path
import cv2
import pickle
import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from PIL import Image
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('./BigGAN/model')

# Prepare a input
truncation = 0.4
class_vector = one_hot_from_names(['lakeshore'], batch_size=1)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

# If you have a GPU, put everything on cuda
# noise_vector = noise_vector.to('cuda')
# class_vector = class_vector.to('cuda')
# model.to('cuda')

# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)

# If you have a GPU put back on CPU
# output = output.to('cpu')



# Save results as png images
save_as_images(output,'./Image/BigGANoutput')
save_as_images(output,'./MiDaS/input/MidasInput')


# resize the Image to 160*256
img = cv2.imread('./MiDaS/input/MidasInput_0.png')
print(img.shape)
height, width = img.shape[:2]
size = (256,160)
img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
print(img.shape)
cv2.imwrite('./MiDaS/input/MidasInput_0.png', img, [int(cv2.IMWRITE_JPEG_QUALITY),95])

#run Midas
os.system('python ./MiDaS/run.py')

#normalize Image

# 读取RGB图片path
input_data_train = './MiDaS/input/MidasInput_0.png'
input_data_train=cv2.imread(input_data_train,cv2.IMREAD_ANYCOLOR)
# 读取视差图path
input_data_disp = './MiDaS/output/MidasInput_0.png'
input_data_disp=cv2.imread(input_data_disp,cv2.IMREAD_ANYCOLOR)

rgb = np.array(input_data_train)
disp = np.array(input_data_disp)

disp = np.array(disp)

rgbd = np.zeros((160, 256, 4), dtype=np.uint8)


rgbd[:, :, 0] = rgb[:, :, 0]
rgbd[:, :, 1] = rgb[:, :, 1]
rgbd[:, :, 2] = rgb[:, :, 2]
rgbd[:, :, 3] = disp
#     input_data_train[i] = os.path.split(input_data_train[i])

#  归一化缩小范围
result=np.zeros(rgbd.shape,np.float32)
cv2.normalize(rgbd, result, 0, 1, cv2.NORM_MINMAX,cv2.CV_32F)
filename = './Image/rgbdImage' + '.png'

#     # 保存的RGBD图片文件名为RGB图片的文件名
cv2.imwrite(filename, rgbd)

#将numpy数组转换为字典
autocruise_input4={'input_rgbd':result}

file = open('./natural/autocruise_input4.pkl', 'wb')
pickle.dump(autocruise_input4, file)
file.close()

#run infinity natural

os.system('cd ./natural;MKL_THREADING_LAYER=GNU;python -m autocruise --output_folder=autocruise --num_steps=100')
print("success")

