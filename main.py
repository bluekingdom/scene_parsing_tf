# -*- coding: utf-8 -*-
"""
* @File Name:           main.py
* @Author:              Wang Yang
* @Created Date:        2017-09-12 20:05:27
* @Last Modified Data:  2017-09-16 20:24:13
* @Desc:                    
*
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import cv2

import os
from  scoreofSegmentation import scoreSegNumpy

from network import ResnetPSPNet
from data_provider import MIT_Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_integer("epoch", "100", "epoch")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Optimizer")
tf.flags.DEFINE_integer("val_per_iter", "100", "how many iter before executing validate")
tf.flags.DEFINE_integer("showlog_pre_iter", "20", "how many iter before show log")
tf.flags.DEFINE_string("checkpoint_file", "", "Path to model checkpoint")
tf.flags.DEFINE_string("data_dir", "/home/blue/data/ADEChallengeData2016/", "Path to data")

def validate(sess, loss, predictions, images, labels, saver, dataProvider, batch_size, image_height, image_width, num_classes):
    val_batches = dataProvider.GetValBatch(batch_size, image_width, image_height)
    val_losses = []
    val_mean_IoU = []
    val_accuracy = []
    val_count = 0
    for val_batch in val_batches:
        if val_count > 10: break

        val_imgs, val_ants = val_batch[0], val_batch[1]

        val_loss,val_prediction = sess.run([loss,predictions], feed_dict={images: val_imgs, labels: val_ants})
        val_losses.append(val_loss)

        mean_IoU,accuracy = scoreSegNumpy(val_prediction, val_ants,num_classes)
        val_mean_IoU.append(mean_IoU)
        val_accuracy.append(accuracy)
        val_count += 1
        pass

    val_loss = np.mean(np.array(val_losses))
    print('val finish, val loss: %0.5f' % (val_loss))
    print('Mean IoU: %0.3f, pixel accuracy: %0.3f %%\n' %(sum(val_mean_IoU)/len(val_mean_IoU),sum(val_accuracy)*100/len(val_accuracy)))

    return val_loss

def train(checkpoint_file=''):
    image_width = 473
    image_height = 473

    dataProvider = MIT_Dataset(FLAGS.data_dir)

    num_classes = dataProvider.GetNumClasses()
    label_weights = dataProvider.GetLabelWeights()

    images = tf.placeholder(tf.float32, [None, image_height, image_width, 3], name="images")
    labels = tf.placeholder(tf.int32, [None, image_height, image_width, 1], name="labels")

    network = ResnetPSPNet()
    logits = network.inference(images, num_classes = num_classes, is_training=True)
    predictions = network.predict(network.inference(images, num_classes, is_training=False, reuse=True), image_height, image_width)
    train_op, loss = network.optimal(logits, labels, FLAGS.learning_rate, label_weights)


    store_variables = tf.trainable_variables()
    # for op in store_variables: print(op.name) # show trainable variables
    saver = tf.train.Saver(store_variables, max_to_keep=1)

    train_iter_step = 0 
    min_val_loss = 1e20 # variable for store minimal validation loss

    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        if checkpoint_file != '':
            saver.restore(sess, checkpoint_file)
            print('restore from checkpoint: ', FLAGS.checkpoint_file)
            pass

        for epoch_idx in range(FLAGS.epoch):
            train_batches = dataProvider.GetTrainBatch(FLAGS.batch_size, image_width, image_height)
            for batch in train_batches:
                imgs = batch[0]
                ants = batch[1]

                # execute validate
                if train_iter_step % FLAGS.val_per_iter == 0:
                    val_loss = validate(sess, loss, predictions, images, labels, saver, dataProvider, FLAGS.batch_size, image_height, image_width, num_classes)

                    # save best model
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        path = saver.save(sess, "models/model", global_step=train_iter_step)
                        print("Saved model checkpoint to {}".format(path))
                        pass

                    # save predict
                    predicted_images = sess.run(predictions, feed_dict={images: imgs, labels: ants})
                    for i in range(imgs.shape[0]):
                        predicted_image, annot_image = predicted_images[i, :, :], ants[i, :, :]

                        cv2.imwrite('images/predict_%d.png' % i, predicted_image)
                        cv2.imwrite('images/annot_%d.png' % i, annot_image)
                        pass
                    pass

                # train
                t1 = time.time()
                l, _ = sess.run([loss, train_op], feed_dict={images: imgs, labels: ants})
                t2 = time.time()

                if train_iter_step % FLAGS.showlog_pre_iter == 0:
                    print('iter idx: %5d, train loss: %0.5f, train time: %0.3f ms' % (train_iter_step, l, (t2-t1)*1000.0/imgs.shape[0]))

                train_iter_step += 1
                pass
            pass

        pass
    pass

if __name__ == '__main__':
    train(FLAGS.checkpoint_file)

