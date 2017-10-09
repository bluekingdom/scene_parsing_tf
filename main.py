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

import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_integer("epoch", "100", "epoch")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Optimizer")
tf.flags.DEFINE_integer("val_per_iter", "100", "how many iter before executing validate")
tf.flags.DEFINE_string("checkpoint_file", "", "Path to model checkpoint")

class DataProvider:
    images = {}
    annots = {}
    input_width = 0
    input_height = 0

    def __init__(self):
        self.images['training'] = []
        self.images['validation'] = []
        self.annots['training'] = []
        self.annots['validation'] = []
        pass

    def Init_Mit_Scene_Parse(self):
        folder_root = '/home/blue/data/ADEChallengeData2016/'

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

    @staticmethod
    def GetBatch(images, annots, batch_size, image_width, image_height, shuffle):

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
                    img = cv2.resize(cv2.imread(img_filename), (image_width, image_height)) 
                    if len(img.shape) < 3:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0 - 128.0
                    imgs.append(img) 

                    ant_filename = annots[i]
                    img = cv2.resize(cv2.imread(ant_filename, cv2.IMREAD_GRAYSCALE), (image_width, image_height)) 
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

        return DataProvider.GetBatch(train_images, train_annots, batch_size, image_width, image_height, shuffle)
        pass


    def GetValBatch(self, batch_size, image_width, image_height, shuffle=True):
        val_images = np.array([image for image in self.images['validation']])
        val_annots = np.array([annot for annot in self.annots['validation']])

        return DataProvider.GetBatch(val_images, val_annots, batch_size, image_width, image_height, shuffle)

        pass

    pass


class Network:
    def root_block(self, inputs, scope=None):
        with tf.variable_scope(scope, 'root', [inputs]) as sc:
            conv1 = slim.conv2d(inputs, 64, [3, 3], stride=2, scope='conv1')
            conv2 = slim.conv2d(conv1,  64, [3, 3], stride=1, scope='conv2')
            conv3 = slim.conv2d(conv2, 128, [3, 3], stride=1, scope='conv3')
            pool1 = slim.max_pool2d(conv3, [3, 3], stride=2, scope='pool1')

        return pool1

    def residual_block(self, inputs, output_num, stride=1, scope=None):
        with tf.variable_scope(scope, 'residual_block', [inputs]) as sc:
            with slim.arg_scope([slim.conv2d], padding='SAME', 
                # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                weights_regularizer = slim.l2_regularizer(0.0005), 
                activation_fn=None):

                depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

                depth_bottleneck = depth_in // 4

                shortcut = slim.conv2d(inputs, output_num, [1, 1], stride=stride,  scope='shortcut')

                residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=stride, scope='conv1')
                residual = slim.batch_norm(residual, scope='bn1')
                residual = tf.nn.relu(residual)
                residual = slim.conv2d(residual, depth_bottleneck, [3, 3], stride=1, scope='conv2')
                residual = slim.batch_norm(residual, scope='bn2')
                residual = tf.nn.relu(residual)
                residual = slim.conv2d(residual, output_num, [1, 1], stride=1, scope='conv3')

                output = tf.nn.relu(shortcut + residual)

        return output
        
    def pyramid_pooling_module(self, inputs, scope = None):
        def branch(inputs, bin_size, name):
            inputs_shape = inputs.get_shape().as_list()
            pool_size = inputs_shape[1] // bin_size
            print('name: %s, shape: %d, bin_size: %d' % (name, inputs_shape[1], bin_size))
            with tf.variable_scope(scope, 'branch_block_%s' % name, [inputs]) as sc:
                with slim.arg_scope([slim.conv2d], padding='SAME', 
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.0005), 
                    activation_fn=None):

                    dims = inputs.get_shape().dims
                    out_height, out_width, depth = dims[1].value, dims[2].value, dims[3].value

                    pool1 = slim.avg_pool2d(inputs, pool_size, stride=pool_size, scope='pool1')
                    conv1 = slim.conv2d(pool1, depth, [1, 1], stride=1, scope='conv1')
                    output = tf.image.resize_bilinear(conv1, [out_height, out_width])

            return output 

        with tf.variable_scope(scope, 'pyramid_pooling_module', [inputs]) as sc:
            branchs = [inputs]
            for bin in [1, 2, 3, 6]:
                b = branch(inputs, bin_size = bin, name = 'branch_bin_%s' % bin)
                branchs.append(b)
            net = tf.concat(axis=3, values=branchs)
            pass

        return net

    def inference(self, inputs, num_classes, is_training=False, reuse=False, scope=None):
        with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:

            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = self.root_block(inputs)
                for p in [(256, 2), (256, 1), (512, 2), (512, 1)]:
                    output_num = p[0]
                    stride = p[1]
                    net = self.residual_block(net, output_num, stride=stride)
                pass
                net = self.pyramid_pooling_module(net)

                net = slim.conv2d(net, 1024, [3, 3], stride=1, scope='fc1')
                net = slim.dropout(net, keep_prob=0.9, is_training=is_training)

                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

                probabilities = slim.softmax(logits, scope='probabilities')

                predictions = tf.argmax(probabilities, -1)
                predictions = tf.expand_dims(predictions, -1)

                dims = inputs.get_shape().dims
                out_height, out_width = dims[1].value, dims[2].value
                predictions = tf.image.resize_bilinear(predictions, [out_height, out_width])
                predictions = tf.squeeze(predictions, squeeze_dims=[3])

        return predictions, logits

    def optimal(self, logits, annotations):

        dims = logits.get_shape().dims
        out_height, out_width = dims[1].value, dims[2].value
        annotations = tf.image.resize_bilinear(annotations, [out_height, out_width])
        annotations = tf.cast(annotations, tf.int32)

        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.squeeze(annotations, squeeze_dims=[3]), name="entropy")))
        total_loss = tf.losses.get_total_loss() + loss
        # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)

        return train_op, total_loss

    def train(self, checkpoint_file=''):
        image_width = 500
        image_height = 500
        num_classes = 151
        images = tf.placeholder(tf.float32, [None, image_height, image_width, 3], name="images")
        labels = tf.placeholder(tf.int32, [None, image_height, image_width, 1], name="labels")

        predictions, logits = self.inference(images, num_classes = num_classes, is_training=True)
        train_op, loss = self.optimal(logits, labels)

        dataProvider = DataProvider()
        dataProvider.Init_Mit_Scene_Parse()

        store_variables = tf.trainable_variables()
        # for op in store_variables: print(op.name) # show trainable variables
        # store_variables = tf.global_variables()
        saver = tf.train.Saver(store_variables, max_to_keep=1)
        # saver = tf.train.Saver(max_to_keep=1)

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
                        val_batches = dataProvider.GetValBatch(FLAGS.batch_size, image_width, image_height)
                        val_losses = []
                        val_count = 0
                        for val_batch in val_batches:
                            if val_count > 10: break

                            val_imgs, val_ants = val_batch[0], val_batch[1]
                            # [val_loss, accuracy_value, mean_IOU_value, _] = sess.run([loss, accuracy, mean_IOU, metrics_op], feed_dict={images: val_imgs, labels: val_ants})
                            val_loss = sess.run(loss, feed_dict={images: val_imgs, labels: val_ants})
                            val_losses.append(val_loss)
                            val_count += 1
                            pass

                        val_loss = np.mean(np.array(val_losses))
                        # print('val finish, val loss: %0.5f, accuracy value: %0.3f, mean iou value: %0.3f' % (val_loss, accuracy_value, mean_IOU_value))
                        print('val finish, val loss: %0.5f' % (val_loss))

                        # save best model
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            path = saver.save(sess, "models/model", global_step=train_iter_step)
                            print("Saved model checkpoint to {}".format(path))
                            pass

                        # save predict
                        predicted_images = sess.run(predictions, feed_dict={images: val_imgs, labels: val_ants})
                        for i in range(val_ants.shape[0]):
                            predicted_image, annot_image = predicted_images[i, :, :], val_ants[i, :, :]


                            cv2.imwrite('images/predict_%d.png' % i, predicted_image)
                            cv2.imwrite('images/annot_%d.png' % i, annot_image)
                            pass
                        pass

                    # train
                    l, _ = sess.run([loss, train_op], feed_dict={images: imgs, labels: ants})
                    print('iter idx: %5d, train loss: %0.5f' % (train_iter_step, l))

                    train_iter_step += 1
                    pass
                pass

            pass
        pass

    pass

if __name__ == '__main__':
    network = Network()
    network.train(FLAGS.checkpoint_file)

