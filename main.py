# -*- coding: utf-8 -*-
"""
* @File Name:   		main.py
* @Author:				Wang Yang
* @Created Date:		2017-09-12 20:05:27
* @Last Modified Data:	2017-09-16 14:57:54
* @Desc:					
*
"""

# import tensorflow as tf
import os
import cv2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer("epoch", "10", "epoch")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")

class DataProvider:
	images = {}
	annots = {}
	input_width = 0
	input_height = 0

	def Init(self, input_width, input_height):
		self.images['training'] = []
		self.images['validation'] = []
		self.annots['training'] = []
		self.annots['validation'] = []
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

		_load("training")
		_load("validation")

		pass

	def GetTrainBatch(self, batch_size):

		len_of_images = len(self.images)

		num_batches_per_epoch = len(len_of_images) // batch_size

	    if (shuffle):
	    	perm = np.arange(len_of_images)
	        np.random.shuffle(perm)
	        self.images = self.images[perm]
	        self.annots = self.annots[perm]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            images = []
            annots = []
            for i in range(start_index, end_index + 1):
                try:
                	img_filename = self.images[i]
                    img = cv2.resize(cv2.imread(img_filename), (image_width, image_height)) 
                    if len(img.shape) < 3:
                    	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    images.append(img) 

                    ant_filename = self.annots[i]
                    img = cv2.resize(cv2.imread(ant_filename), (image_width, image_height)) 
                    annots.append(img)

                except Exception as err:
                    print(err) 
                    # batch = [cv2.resize(cv2.imread(d), (x_h, x_w)) for d in data[start_index:end_index]]

            if len(images) == 0:
                continue

            yield np.array(images), np.array(annots)


	def GetTestBatch(self, batch_size, image_width, image_height):
		for i in range(batch_size):

			pass
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
		return logits, predict

	def optimal(logits, labels):
	    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
	    	logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="entropy")))
	    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	    return optimizer, loss

	def train():
		image_width = 300
		image_width = 300
		channels = 3
		images = tf.placeholder("float", [None, image_height, image_width, channels])
		labels = tf.placeholder("float", [None, image_height, image_width, channels])

		logits, predict = self.inference(images)
		optimizer, loss = self.optimal(logits, labels)

        train_batch_idx = 0

		with tf.Session as sess:
			tf.global_variables_initializer().run()

			for epoch_idx in range(FLAGS.epoch):
				train_batches = self.GetTrainBatch(image_width, image_height)
				for batch in train_batches:
					l = sess.run([loss, optimizer], feed_dict={images: batch, labels: })
				print('iter idx: %5d, train loss: %0.5f' % (train_batch_idx, l))
				train_batch_idx += 1
				pass

			pass
		pass

	pass

if __name__ == '__main__':
	train()
