from __future__ import division
import numpy as np
import tensorflow as tf
from collections import deque
import random
import string
import tensorflow.contrib.slim as slim

np.random.seed(1)
tf.set_random_seed(1)


class AC_net:

    def __init__(self, scope, trainer, x_dim_p, y_dim_p, z_dim_p, n_actions, x_dim_v, y_dim_v, z_dim_v, filter_size,
                 num_filters, num_residuals_policy, regu_scalar, num_filters_last_cov):
        self.scope = scope
        self.x_dim_p = x_dim_p
        self.y_dim_p = y_dim_p
        self.z_dim_p = z_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.y_dim_v = y_dim_v
        self.z_dim_v = z_dim_v
        self.filter_sizeH = filter_size
        self.filter_sizeV = filter_size
        self.num_filters = num_filters
        self.num_residuals_policy = num_residuals_policy
        self.regu_scalar = regu_scalar
        self.num_filters_last_cov = num_filters_last_cov
        self.padding = 'VALID'  # 'SAME'#

        self.regularizer = tf.contrib.layers.l2_regularizer(self.regu_scalar, scope=None)

        self.w_initializer, self.b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        self.Input_p = tf.placeholder(tf.float32, [None, self.x_dim_p, self.y_dim_p, self.z_dim_p], name='policy_input')
        self.Input_v = tf.placeholder(tf.float32, [None, self.x_dim_v, self.y_dim_v, self.z_dim_v], name='value_input')

        with tf.variable_scope(self.scope):
            with tf.variable_scope('policy', regularizer=self.regularizer):
                hidden_p = self.dnn(self.Input_p, x_dim_p, y_dim_p, z_dim_p)

                # self.policy = tf.layers.dense(hidden_p, self.n_actions, activation = tf.nn.softmax, kernel_initializer = self.w_initializer, bias_initializer = self.b_initializer) # b_initializer = None??
                self.policy = slim.fully_connected(hidden_p, self.n_actions,
                                                   activation_fn=tf.nn.softmax,
                                                   weights_initializer=self.normalized_columns_initializer(0.01),
                                                   biases_initializer=None)

            with tf.variable_scope('value', regularizer=self.regularizer):
                hidden_v = self.dnn(self.Input_v, x_dim_v, y_dim_v, z_dim_v)

                # self.value = tf.layers.dense(hidden_v, 1, activation = None, kernel_initializer = self.w_initializer, bias_initializer = self.b_initializer)
                self.value = slim.fully_connected(hidden_v, 1,
                                                  activation_fn=None,
                                                  weights_initializer=self.normalized_columns_initializer(1.0),
                                                  biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if self.scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, self.n_actions, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.regu_loss_policy = tf.add_n(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope + '/policy'))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-6))  # 1e-6 is for preventing NaN
                lost_policy_net = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-6) * self.advantages)
                self.loss_policy = lost_policy_net - self.entropy * 0.1 + self.regu_loss_policy

                self.regu_loss_value = tf.add_n(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope + '/value'))
                loss_value_net = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.loss_value = loss_value_net + self.regu_loss_value

                # Get gradients from local network using local losses
                local_vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/policy')
                self.gradients_policy = tf.gradients(self.loss_policy, local_vars_policy)
                self.var_norms_policy = tf.global_norm(local_vars_policy)
                grads_policy, self.grad_norms_policy = tf.clip_by_global_norm(self.gradients_policy, 40.0)

                local_vars_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/value')
                self.gradients_value = tf.gradients(self.loss_value, local_vars_value)
                self.var_norms_value = tf.global_norm(local_vars_value)
                grads_value, self.grad_norms_value = tf.clip_by_global_norm(self.gradients_value, 40.0)

                # Apply local gradients to global network
                global_vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/policy')
                self.apply_grads_policy = trainer.apply_gradients(zip(grads_policy, global_vars_policy))

                global_vars_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/value')
                self.apply_grads_value = trainer.apply_gradients(zip(grads_value, global_vars_value))

    '''def dnn(self, InputFeatures, x_dim, y_dim, z_dim):
        with tf.variable_scope('first_conv'): # initiate with a convolution block
            x_h = self._conv(InputFeatures, [self.filter_sizeV, self.filter_sizeH, z_dim, self.num_filters], 1) # [filter_height, filter_width, in_channels, out_channels]
            x_h = self._batch_norm(x_h, self.num_filters)
            x_h = tf.nn.elu(x_h) 
        for ii in range(self.num_residuals_policy): # residual blocks
            with tf.variable_scope('residual_%d' % ii):
                x_h = self._residual_unit(x_h, self.num_filters, self.num_filters, 1) # stride = 1
        with tf.variable_scope('head_conv'): # anotoher convolution block
            x_h = self._conv(x_h, [1, 1, self.num_filters, self.num_filters_last_cov], 1)
            x_h = self._batch_norm(x_h, self.num_filters_last_cov)
            x_h = tf.nn.elu(x_h)  
        with tf.variable_scope('head_fc1'): # fully-connected layer
            x_h = slim.flatten(x_h)
            #x_h = tf.layers.dense(x_h, 128, tf.nn.elu, kernel_initializer = self.w_initializer, bias_initializer = self.b_initializer)
            x_h = slim.fully_connected(x_h, 128, activation_fn = tf.nn.elu)
        return x_h'''

    def dnn(self, InputFeatures, x_dim, y_dim, z_dim):
        with tf.variable_scope('first_conv'):
            x_h = slim.conv2d(activation_fn=tf.nn.elu,
                              inputs=InputFeatures, num_outputs=self.num_filters,
                              kernel_size=[self.filter_sizeV, self.filter_sizeH], stride=[1, 1], padding=self.padding)
        for ii in range(self.num_residuals_policy * 2):
            with tf.variable_scope('cov_%d' % ii):
                x_h = slim.conv2d(activation_fn=tf.nn.elu,
                                  inputs=x_h, num_outputs=self.num_filters,
                                  kernel_size=[self.filter_sizeV, self.filter_sizeH], stride=[1, 1],
                                  padding=self.padding)
        with tf.variable_scope('head_conv'):
            x_h = slim.conv2d(activation_fn=tf.nn.elu,
                              inputs=x_h, num_outputs=self.num_filters_last_cov,
                              kernel_size=[1, 1], stride=[1, 1], padding=self.padding)
        with tf.variable_scope('head_fc1'):
            x_h = slim.flatten(x_h)
            x_h = slim.fully_connected(x_h, 128, activation_fn=tf.nn.elu)
        return x_h

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def _conv(self, input_, filter_shape, stride):  # Convolutional layer
        '''return tf.nn.conv2d(input,
                            filter = self.init_rand_tensor(filter_shape),
                            strides = [1, stride, stride, 1],
                            padding = self.padding)'''
        return tf.layers.conv2d(inputs=input_,
                                filters=filter_shape[3],
                                kernel_size=[filter_shape[0], filter_shape[1]],
                                strides=[stride, stride],
                                padding=self.padding,
                                activation=tf.nn.elu)

    def _batch_norm(self, input_,
                    out_channels):  # 'out_channels' equates to the deepth of 'input_' ([batch, height, width, depth])
        # batch normalization
        mean, var = tf.nn.moments(input_, axes=[0, 1,
                                                2])  # for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2]
        offset = tf.Variable(tf.zeros([out_channels]))
        scale = tf.Variable(tf.ones([out_channels]))
        batch_norm = tf.nn.batch_normalization(input_, mean, var, offset, scale,
                                               0.001)  # 0.001: small float number to avoid dividing by 0
        return batch_norm

    def _residual_unit(self, input_, in_filters, out_filters, stride):
        # Residual unit with 2 sub-layers, option 0: zero padding

        if self.padding == 'SAME':
            option = 0
        else:
            option = 1

        # first convolution layer
        x = self._conv(input_, [self.filter_sizeV, self.filter_sizeH, in_filters, out_filters], stride)
        x = self._batch_norm(x, out_filters)
        x = tf.nn.elu(x)

        # second convolution layer
        x = self._conv(x, [self.filter_sizeV, self.filter_sizeH, out_filters, out_filters], stride)
        x = self._batch_norm(x, out_filters)

        if in_filters != out_filters:
            if option == 0:
                difference = out_filters - in_filters
                left_pad = difference // 2
                right_pad = difference - left_pad
                identity = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
                return tf.nn.elu(x + identity)
            else:
                exit(1)
        else:
            return tf.nn.elu(x + input_)

    '''def init_rand_tensor(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0)) # output random data following the normal distribution'''