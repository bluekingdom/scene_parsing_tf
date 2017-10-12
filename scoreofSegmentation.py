# -*- coding: utf-8 -*-
"""
* @File Name:           scoreofSegmentation.py
* @Author:              LI Yanlin
* @Created Date:        2017-10-09 15:03:11
* @Last Modified Data:  2017-10-10 10:16:38
* @Desc:                    
*
"""
'''
segofSegmentation.py包含三个主要functions，分别是：
IoU,accurary=scoreSegPath(path,num_class):针对路径中pred和anno文件夹中的结果进行对比
scoreSegSingle(imPred,imAnno,num_class)：对单张结果分析
scoreSegNumpy(npPred,npAnno,num_class)：对数组结果分析
'''
'''
This function is for the score of sementic segmentation like IoU and 
pixel-wise accuracy and finally return a score
'''

import cv2
import numpy as np
import sys
import os
from math import fsum


####################################################################################
#IoU
def intersectionAndUnion(imPred,imLab,num_class):
    #imPred = imPred*(imLab>0);
    intersection = imPred*(imPred==imLab)
    area_intersection,bins=np.histogram(intersection,bins=range(num_class+1), range=(0,num_class+1))
    area_pred,bins = np.histogram(imPred,bins=range(num_class+1), range=(0,num_class+1))
    area_lab,bins = np.histogram(imLab,bins=range(num_class+1), range=(0,num_class+1))
    area_union = area_pred + area_lab - area_intersection

    area_intersection = area_intersection[1:]
    area_union = area_union[1:]
    return area_intersection,area_union

#PixelAccuracy
def pixelAccuracy(imPred,imLab):
    pixel_labeled = np.sum(imLab>0)
    pixel_correct = np.sum((imPred==imLab)*(imLab>0))
    pixel_accuracy = pixel_correct / pixel_labeled

    return pixel_correct,pixel_labeled,pixel_accuracy

####################################################################################
#ScoreSegPath
def scoreSegPath(path,num_class):

    pathPred = "%s%s"%(path,"/Pred/")
    pathAnno = "%s%s"%(path,"/Anno/")

    # initialize statistics
    cnt=0
    area_intersection = []
    area_union = []
    pixel_accuracy = []
    pixel_correct = []
    pixel_labeled = []

    files = [file for file in os.listdir(pathPred)]
    for file in files:
      anno_file = pathAnno + file
      pred_file = pathPred + file
      if False == os.path.exists(anno_file):
        print('file not exists: %s' % (anno_file))
        continue
      imPred = cv2.imread(pred_file)
      imAnno = cv2.imread(anno_file)
      # check image size
      if imPred.shape!=imAnno.shape:
        #print('Label image [%s] should have the same size as label image! Resizing...\n'%(fileLab))
        #np.reshape(imPred.shape,imAnno.shape)
        imPred=np.expand_dims(imPred, axis=2)     
      # compute IoU
      #print('Evaluating %d/%d...\n'%(cnt)%(len(files))
      ai,au= intersectionAndUnion(imPred, imAnno, num_class)
      area_intersection.append(ai),area_union.append(au)
      #compute pixel-wise accuracy
      pc,pl,pa= pixelAccuracy(imPred, imAnno)
      pixel_correct.append(pc), pixel_labeled.append(pl),pixel_accuracy.append(pa),  
      cnt = cnt + 1
    pass

    #Summary
    sau=sum(area_union)
    for i in range(len(sau)) :
      if sau[i]==0:sau[i]=1
    sau=sau.astype(np.float64)
    IoU = sum(area_intersection)/sau
    mean_IoU = sum(IoU)/len([b for b in IoU if b>0])

    accuracy = float(sum(pixel_correct))/sum(pixel_labeled)
    #print('==== Summary ====\n')
    #print('Mean IoU over %d classes: %f \n' %(num_class,mean_IoU))
    #print('Pixel-wise Accuracy: %f %% \n' %(accuracy*100))
    return mean_IoU,accuracy
####################################################################################
#ScoreSegSingle
def scoreSegSingle(imPred,imAnno,num_class):
    
    
    # check image size
    if imPred.shape!=imAnno.shape:
      #print('Label image should have the same size as label image! Resizing...\n')
      #np.reshape(imPred.shape,imAnno.shape)
      imPred=np.expand_dims(imPred, axis=2)
             
    # compute IoU
    area_intersection,area_union= intersectionAndUnion(imPred, imAnno, num_class)
    #compute pixel-wise accuracy
    pixel_correct, pixel_labeled,pixel_accuracy = pixelAccuracy(imPred, imAnno)
       
    #Summary
    sau=area_union
    for i in range(len(sau)) :
      if sau[i]==0:sau[i]=1
    sau=sau.astype(np.float64)
    IoU = area_intersection/sau
    mean_IoU = sum(IoU)/len([b for b in IoU if b>0])

    accuracy = float(pixel_correct)/pixel_labeled
    #print('==== Summary ====\n')
    #print('Mean IoU over %d classes: %f \n' %(num_class,mean_IoU))
    #print('Pixel-wise Accuracy: %f %% \n' %(accuracy*100))
    return mean_IoU,accuracy
####################################################################################
def scoreSegNumpy(npPred,npAnno,num_class):
# initialize statistics
    cnt=0
    area_intersection = []
    area_union = []
    pixel_accuracy = []
    pixel_correct = []
    pixel_labeled = []
    if npPred.shape!=npAnno.shape:
      npPred=np.expand_dims(npPred, axis=3)

    for i in range(npPred.shape[0]):
      imPred=npPred[i]
      imAnno=npAnno[i]
      # check image size

             
      # compute IoU
      #print('Evaluating %d/%d...\n'%(cnt)%(len(files))
      ai,au= intersectionAndUnion(imPred, imAnno, num_class)
      area_intersection.append(ai),area_union.append(au)
      #compute pixel-wise accuracy
      pc,pl,pa= pixelAccuracy(imPred, imAnno)
      pixel_correct.append(pc), pixel_labeled.append(pl),pixel_accuracy.append(pa),  
      cnt = cnt + 1
    pass

    #Summary
    sau=sum(area_union)
    for i in range(len(sau)) :
      if sau[i]==0:sau[i]=1
    sau=sau.astype(np.float64)
    IoU = sum(area_intersection)/sau
    mean_IoU = sum(IoU)/len([b for b in IoU if b>0])

    accuracy = float(sum(pixel_correct))/sum(pixel_labeled)
    #print('==== Summary ====\n')
    #print('Mean IoU over %d classes: %f \n' %(num_class,mean_IoU))
    #print('Pixel-wise Accuracy: %f %% \n' %(accuracy*100)) 
    return mean_IoU,accuracy