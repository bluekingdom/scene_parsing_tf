# -*- coding: utf-8 -*-
"""
* @File Name:   		preprocess.py
* @Author:				Wang Yang
* @Created Date:		2017-09-13 10:56:39
* @Last Modified Data:	2017-09-13 10:56:58
* @Desc:					
*
"""


import tensorflow as tf
import scipy.io


data = scipy.io.loadmat("/home/blue/data/ade20k/ADE20K_2016_07_26/index_ade20k.mat")


for d in data:
	print(d)
	print(data[d])
# print(data)
