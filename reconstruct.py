#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:53:25 2017

@author: shijie
"""
import scipy.misc
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import myfun

img_o = scipy.misc.imread('test/baboon.bmp',flatten=True, mode='YCbCr').astype(np.float)
img = myfun.create_LR(img_o,4)
img_size = 32
stride = 16
h,w = img.shape
piece_wise = []
for x in range(0, h-img_size+1, stride):
    for y in range(0, w-img_size+1, stride):
        sub_input = img[x:x+img_size, y:y+img_size].reshape(img_size,img_size,1) # [32 x 32]
        piece_wise.append(sub_input)
input_ = np.asarray(piece_wise)        
srcnn = load_model('SRCNN_model.h5')
hat = srcnn.predict(input_)
img_re = np.zeros(img.shape)
i=0
for x in range(0, h-img_size+1, stride):
    for y in range(0, w-img_size+1, stride):
        img_re[x:x+img_size, y:y+img_size] = hat[i].reshape(img_size,img_size)
        i += 1
#img_re = img_re
#img_rgb = myfun.color_convert(img_re.astype(np.uint8))
plt.imshow(img_re)