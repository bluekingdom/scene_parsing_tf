# -*- coding: utf-8 -*-
"""
* @File Name:   		data_provider.py
* @Author:				Wang Yang
* @Created Date:		2017-10-10 13:37:32
* @Last Modified Data:	2017-10-12 10:31:58
* @Desc:					
*
"""

import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import cPickle as pickle

# --
def save_var(var_name, var, temp_folder='./temp'):
    if False == os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    var_file_path = '%s/%s.pkl' %(temp_folder, var_name)
    fp = open(var_file_path, 'w+')
    pickle.dump(var, fp, -1)
    fp.close()
    print('save_var: ' + var_file_path)
    return var_file_path

def load_var(var_name, temp_folder='./temp'):
    var_file_path = '%s/%s.pkl' %(temp_folder, var_name)
    if False == os.path.exists(var_file_path):
        print('var file path not exist: ' + var_file_path)
        return None

    fp = open(var_file_path)
    var = pickle.load(fp)
    fp.close()
    print('load_var: ' + var_file_path)
    return var

# --


class DataProvider:
    images = {}
    annots = {}
    label_weights = []

    def __init__(self, folder_root):
        self.images['training'] = []
        self.images['validation'] = []
        self.annots['training'] = []
        self.annots['validation'] = []

        if os.path.exists(folder_root) == False:
            print('data dir not exist: %s' % folder_root)
            sys.exit()

        self.Init(folder_root)
        self.label_weights = self.CalcLabelWeights()
        pass

    def Init(self, folder_root):
        raise NameError('must be override!')

    def LoadImg(self, filename):
        raise NameError('must be override!')

    def LoadAnnot(self, filename):
        raise NameError('must be override!')

    def GetNumClasses(self):
        raise NameError('must be override!')

    def CalcLabelWeights(self):
        label_weights = load_var('label_weights', './%s' % self.__class__.__name__)

        if label_weights == None:
            print('begin to calc label weight!')

            imgs_file = self.annots['training'] + self.annots['validation']

            label_count = [0.0] * 256
            pixel_count = 0

            for img_file in tqdm(imgs_file):
                img = self.LoadAnnot(img_file)

                for r in range(img.shape[0]):
                    for c in range(img.shape[1]):
                        label_count[img[r][c]] += 1

                pixel_count += img.shape[0] * img.shape[1]
                pass

            label_weights = [(c / pixel_count) for c in label_count if c > 0]
            save_var('label_weights', label_weights, './%s' % self.__class__.__name__)
            pass

        label_weights = [(1 / w) for w in label_weights]
        m = sum(label_weights) / len(label_weights)
        label_weights = [w / m for w in label_weights]

        return label_weights

    def GetLabelWeights(self):
        return self.label_weights
 
    def GetBatch(self, images, annots, batch_size, image_width, image_height, shuffle):

        len_of_images = len(images)

        num_batches_per_epoch = len_of_images // batch_size

        if (shuffle):
            perm = np.arange(len_of_images)
            np.random.shuffle(perm)
            images = images[perm]
            annots = annots[perm]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len_of_images)
            imgs = []
            ants = []
            for i in range(start_index, end_index):
                try:
                    img_filename = images[i]
                    img = self.LoadImg(img_filename)
                    img = cv2.resize(img, (image_width, image_height)) 
                    if len(img.shape) < 3:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = (img.astype(np.float32)-128.0) /128.0
                    imgs.append(img) 

                    ant_filename = annots[i]
                    img = self.LoadAnnot(ant_filename)
                    img = cv2.resize(img, (image_width, image_height)) 
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    ants.append(img)

                except Exception as err:
                    print(err) 
                    # batch = [cv2.resize(cv2.imread(d), (x_h, x_w)) for d in data[start_index:end_index]]

            if len(imgs) == 0:
                continue

            yield np.array(imgs), np.expand_dims(np.array(ants), axis=3)
        pass

    def GetTrainBatch(self, batch_size, image_width, image_height, shuffle=True):

        train_images = np.array([image for image in self.images['training']])
        train_annots = np.array([annot for annot in self.annots['training']])

        return self.GetBatch(train_images, train_annots, batch_size, image_width, image_height, shuffle)
        pass


    def GetValBatch(self, batch_size, image_width, image_height, shuffle=True):
        val_images = np.array([image for image in self.images['validation']])
        val_annots = np.array([annot for annot in self.annots['validation']])

        return self.GetBatch(val_images, val_annots, batch_size, image_width, image_height, shuffle)
    pass


class MIT_Dataset(DataProvider):
    def __init__(self, folder_root):
        # super(self.__class__, self).__init__()
        DataProvider.__init__(self, folder_root)

    def LoadImg(self, filename):
        return cv2.imread(filename)

    def LoadAnnot(self, filename):
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    def GetNumClasses(self):
        return 151

    def Init(self, folder_root):
        # folder_root = '/home/blue/data/ADEChallengeData2016/'

        def _load(phase):
            images_root = folder_root + 'images/' + phase + '/'
            annots_root = folder_root + 'annotations/' + phase + '/'

            files = [file for file in os.listdir(images_root)]

            for file in files:
                annot_file = annots_root + file.replace('.jpg', '.png')
                if False == os.path.exists(annot_file):
                    print('file not exists: %s' % (annot_file))
                    continue
                self.images[phase].append(images_root + file)
                self.annots[phase].append(annot_file)
            pass

        _load("training")
        _load("validation")

        pass
    pass

class Synthia_Dataset(DataProvider):
    def __init__(self, folder_root):
        DataProvider.__init__(self, folder_root)

    def LoadAnnot(self):
        pass

    def LoadImg(self):
        pass

    def Init(self, folder_root):
        pass

    def GetNumClasses(self):
        pass
