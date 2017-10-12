# -*- coding: utf-8 -*-
"""
* @File Name:           network.py
* @Author:              Wang Yang
* @Created Date:        2017-10-10 13:18:28
* @Last Modified Data:  2017-10-10 14:18:37
* @Desc:                    
*
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Network:
    def inference(self, inputs, num_classes, is_training=False, reuse=False):
        raise NameError('must be overrided!')
        pass

    def predict(self, logits, image_height, image_width):
        raise NameError('must be overrided!')
        pass

    def optimal(self, logits, annotations, learning_rate):
        raise NameError('must be overrided!')
        pass


class ResnetPSPNet(Network):
    def __root_block(self, inputs, scope=None):
        # with tf.variable_scope(scope, 'root', [inputs]) as sc:
        with tf.variable_scope(scope, 'root') as sc:
            with slim.arg_scope([slim.conv2d], padding='SAME', 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                # weights_initializer = tf.contrib.layers.xavier_initializer(),
                weights_regularizer = slim.l2_regularizer(0.0005), 
                activation_fn=None):
                net = slim.conv2d(inputs, 64, [3, 3], stride=2, scope='conv1')
                net = slim.batch_norm(net, scope='bn1')
                net = tf.nn.relu(net)
                net = slim.conv2d(net,  64, [3, 3], stride=1, scope='conv2')
                net = slim.batch_norm(net, scope='bn2')
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')
                net = slim.batch_norm(net, scope='bn3')
                net = tf.nn.relu(net)
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
        return net

    def __residual_block(self, inputs, output_num, stride=1, dilate_rate=1, scope=None):
        # with tf.variable_scope(scope, 'residual_block', [inputs]) as sc:
        with tf.variable_scope(scope, 'residual_block') as sc:
            with slim.arg_scope([slim.conv2d], padding='SAME', 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                # weights_initializer = tf.contrib.layers.xavier_initializer(),
                weights_regularizer = slim.l2_regularizer(0.0005), 
                activation_fn=None):

                depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

                depth_bottleneck = depth_in // 4

                if output_num == depth_in and stride == 1:
                    shortcut = inputs
                else:
                    shortcut = slim.conv2d(inputs, output_num, [1, 1], stride=stride,  scope='shortcut-conv')
                    shortcut = slim.batch_norm(shortcut, scope='shortcut-bn')

                residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=stride, scope='conv1')
                residual = slim.batch_norm(residual, scope='bn1')
                residual = tf.nn.relu(residual)

                residual = slim.conv2d(residual, depth_bottleneck, [3, 3], stride=1, rate=dilate_rate, scope='conv2')
                residual = slim.batch_norm(residual, scope='bn2')
                residual = tf.nn.relu(residual)

                residual = slim.conv2d(residual, output_num, [1, 1], stride=1, scope='conv3')
                residual = slim.batch_norm(residual, scope='bn3')

                output = tf.nn.relu(shortcut + residual)

        return output

    def __pyramid_pooling_module(self, inputs, scope = None):
        def branch(inputs, bin_size, name):
            inputs_shape = inputs.get_shape().as_list()
            pool_size = inputs_shape[1] // bin_size
            print('name: %s, shape: %d, bin_size: %d' % (name, inputs_shape[1], bin_size))
            # with tf.variable_scope(scope, 'branch_block_%s' % name, [inputs]) as sc:
            with tf.variable_scope(scope, 'branch_block_%s' % name) as sc:
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME', 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    # weights_initializer = tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.0005), 
                    activation_fn=None):

                    dims = inputs.get_shape().dims
                    out_height, out_width, depth = dims[1].value, dims[2].value, dims[3].value

                    pool1 = slim.avg_pool2d(inputs, pool_size, stride=pool_size, scope='pool1')
                    conv1 = slim.conv2d(pool1, depth, [1, 1], stride=1, scope='conv1')
                    bn1 = slim.batch_norm(conv1, scope='bn1')
                    relu1 = tf.nn.relu(bn1, name='relu1')

                    output = tf.image.resize_bilinear(relu1, [out_height, out_width])

                    # output = slim.conv2d_transpose(relu1, depth, [3, 3], [pool_size, pool_size])

            return output 

        # with tf.variable_scope(scope, 'pyramid_pooling_module', [inputs]) as sc:
        with tf.variable_scope(scope, 'pyramid_pooling_module') as sc:
            branchs = [inputs]
            for bin in [1, 2, 3, 6]:
                b = branch(inputs, bin_size = bin, name = 'branch_bin_%s' % bin)
                branchs.append(b)
            net = tf.concat(axis=3, values=branchs)
            pass

        return net

    # def inference(self, inputs, num_classes, is_training=False, reuse=False, scope=None):
    def inference(self, inputs, num_classes, is_training=False, reuse=False):
        # with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:
        with tf.variable_scope('pspnet_v1', reuse=reuse) as sc:

            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = self.__root_block(inputs)
                params = [
                    {'output_num': 128, 'stride': 1, 'dilate_rate': 1}, 
                    {'output_num': 256, 'stride': 2, 'dilate_rate': 1}, 
                    {'output_num': 256, 'stride': 1, 'dilate_rate': 1}, 
                    {'output_num': 512, 'stride': 2, 'dilate_rate': 1}, 
                    {'output_num': 512, 'stride': 1, 'dilate_rate': 2}, 
                    {'output_num': 512, 'stride': 1, 'dilate_rate': 4}, 
                ]

                for p in params:
                    output_num = p['output_num']
                    stride = p['stride']
                    dilate_rate = p['dilate_rate']

                    net = self.__residual_block(net, output_num, stride, dilate_rate)
                pass

                net = self.__pyramid_pooling_module(net)
                net = slim.dropout(net, keep_prob=0.5, is_training=is_training)

                net = slim.conv2d(net, 1024, [3, 3], stride=1, scope='fc1')
                net = slim.dropout(net, keep_prob=0.5, is_training=is_training)

                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

        return logits

    def predict(self, logits, image_height, image_width):
        probabilities = slim.softmax(logits, scope='probabilities')

        predictions = tf.argmax(probabilities, -1)
        predictions = tf.expand_dims(predictions, -1)

        # dims = inputs.get_shape().dims
        # out_height, out_width = dims[1].value, dims[2].value
        out_height, out_width = image_height, image_width

        predictions = tf.image.resize_bilinear(predictions, [out_height, out_width])
        predictions = tf.squeeze(predictions, squeeze_dims=[3])

        return predictions

    def optimal(self, logits, annotations, learning_rate, label_weights):

        dims = logits.get_shape().dims
        num_classes = len(label_weights)
        out_height, out_width = dims[1].value, dims[2].value
        annotations = tf.image.resize_bilinear(annotations, [out_height, out_width])
        annotations = tf.cast(annotations, tf.int32)
        annotations = tf.squeeze(annotations, squeeze_dims=[3])
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)
        weights = annotations_ohe * label_weights
        weights = tf.reduce_sum(weights, 3)
        # tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=annotations_ohe, weights=weights, scope="entropy", reduction=tf.losses.Reduction.MEAN)
        tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=annotations_ohe, weights=1.0, scope="entropy", reduction=tf.losses.Reduction.MEAN)
        total_loss = tf.losses.get_total_loss() 
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        return train_op, total_loss
