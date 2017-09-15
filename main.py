# -*- coding: utf-8 -*-
"""
* @File Name:   		main.py
* @Author:				Wang Yang
* @Created Date:		2017-09-12 20:05:27
* @Last Modified Data:	2017-09-14 13:21:26
* @Desc:					
*
"""

import tensorflow as tf
import os

class DataProvider:
	images = {}
	annots = {}

	def Init(self):
		images['training'] = []
		images['validation'] = []
		annots['training'] = []
		annots['validation'] = []
		pass

	def Init_Mit_Scene_Parse(self):
		folder_root = ''
		train_images_root = folder_root + '/' + 'images/training/'
		train_annots_root = folder_root + '/' + 'annotations/training/'

		def _load(phase):
			images_root = folder_root + '/' + 'images/' + phase + '/'
			annots_root = folder_root + '/' + 'annotations/' + phase + '/'

			files = [file for file in os.listdir(images_root)]

			for i in range(files):
				file = files[i]
				if False == os.path.exists(train_annots_root + file):
					continue
				self.images[phase] = train_images_root + file
				self.annots[phase] = train_annots_root + file







		pass

	def GetBatch(self, batch_size):
		pass
	pass


class Network:
	def root_block(inputs, scope=None):
		# conv     3x3, out depth  64, stride 2
		# conv     3x3, out depth  64, stride 1
		# conv     3x3, out depth 128, stride 1
		# max-pool 3x3, out depth 128, stride 2
		with tf.variable_scope(scope, 'root', [inputs]) as sc:
			# depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
			conv1 = slim.conv2d(inputs, 64, [3, 3], stride=2, scope='conv1')
			conv2 = slim.conv2d(conv1,  64, [3, 3], stride=1, scope='conv2')
			conv3 = slim.conv2d(conv2, 128, [3, 3], stride=1, scope='conv3')
			pool1 = slim.max_pool2d(conv3, [3, 3], stride=2, scope='pool1')

		return pool1

	def residual_block(net, output_num, stride=1, scope=None):
		with tf.variable_scope(scope, 'residual_block', [net]) as sc:
			with slim.arg_scope([slim.ops.conv2d], padding='SAME', stddev=0.01, weight_decay=0.0005, activation_fn=None):
				depth_bottleneck = depth_in // 4

				depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

				shortcut = slim.ops.conv2d(net, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')

				residual = slim.ops.conv2d(residual, depth_bottleneck, [1, 1], stride=stride, scope='conv1')
				residual = slim.ops.conv2d(residual, depth_bottleneck, [3, 3], stride=1, scope='conv2')
				residual = slim.conv2d(residual, depth, [1, 1], stride=1, scope='conv3')

				output = tf.nn.relu(shortcut + residual)

		return output
		# return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)
	def pyramid_pooling_module(inputs, scope = None):
		def branch(inputs, bin_size, name):
			inputs_shape = inputs.get_shape().as_list()
			pool_size = inputs_shape[1] // bin_size
			with tf.variable_scope(scope, 'branch_block_%s' % name, [inputs]) as sc:
			    dims = inputs.get_shape().dims
			    out_height, out_width = dims[1].value, dims[2].value

			    pool1 = slim.avg_pool2d(inputs, pool_size, stride=pool_size, scope='pool1')
			    conv1 = slim.conv2d(pool1, depth, [1, 1], stride=1, scope='conv1')
			    output = tf.image.resize_bilinear(conv1, [out_height, out_width])

			return output 

		with tf.variable_scope(scope, 'pyramid_pooling_module', [inputs]) as sc:
			branchs = [inputs]
			for bin in [1, 2, 3, 6]:
				branch = branch(inputs, bin_size = bin, name = 'branch_bin_%s' % bin)
				branchs.append(branch)
		    net = tf.concat(axis=3, values=branchs)
			pass

		return net

	def inference(inputs):
		with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:

		    end_points_collection = sc.name + '_end_points'
			with slim.arg_scope([slim.batch_norm], is_training=is_training):
				net = self.root_block(inputs)
				for i in range([64, 128, 256, 512]):
					net = self.residual_block(net, i, stride=2)
				pass
				net = self.pyramid_pooling_module(net)

		        net = slim.conv2d(net, 512, [3, 3], stride=1, scope='fc1')
		        net = slim.dropout(net, keep_prob=0.9, is_training=is_training)

		        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
		                          normalizer_fn=None, scope='logits')

		        dims = inputs.get_shape().dims
		        out_height, out_width = dims[1].value, dims[2].value
		        net = tf.image.resize_bilinear(net, [out_height, out_width])

				predict = slim.softmax(net, scope='predictions')

		return net, predict

	def train():
		pass

	pass
