from __future__ import division
from collections import defaultdict
import struct
import types
import string
import numpy as np
import os
import math
import copy
import random
import datetime
from Testt2.AC_net import AC_net
import threading
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
from collections import defaultdict


class NetRec_Agent():

    def __init__(self, id, trainer, gamma, episode_size, batch_size, n_actions, NODE_NUM,
                 filter_size, num_filters, num_residuals_policy, regu_scalar, num_filters_last_cov):
        self.name = 'agent_' + str(id)
        self.trainer = trainer
        self.gamma = gamma # penalty on future reward
        self.episode_size = episode_size # number of requests in each episode
        self.batch_size = batch_size
        # self.global_episodes = global_episodes

        # self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + self.name)
        #
        self.x_dim_p = NODE_NUM
        self.y_dim_p = NODE_NUM
        self.z_dim_p = 2
        self.x_dim_v = NODE_NUM
        self.y_dim_v = NODE_NUM
        self.z_dim_v = 2
        self.n_actions = n_actions

        self.local_network = AC_net(scope = self.name,
                                    trainer = self.trainer,
                                    x_dim_p = self.x_dim_p,
                                    y_dim_p = self.y_dim_p,
                                    z_dim_p = self.z_dim_p,
                                    x_dim_v = self.x_dim_v,
                                    y_dim_v = self.y_dim_v,
                                    z_dim_v = self.z_dim_v,
                                    n_actions = self.n_actions,
                                    filter_size=filter_size,
                                    num_filters=num_filters,
                                    num_residuals_policy=num_residuals_policy,
                                    regu_scalar=regu_scalar,
                                    num_filters_last_cov=num_filters_last_cov)
        #self.update_local_ops = self.update_target_graph('global',self.name)

        self.netflag = float(99) # network topology type flag
        self.delayflag = float(9999) # delay change flag
        self.numline = float(0) # Delay.txt len(f.readlines)
        self.numline2 = float(0)  # Thrpughput.txt len(f.readlines)
        self.numline3 = float(0)  # PktLoss.txt len(f.readlines)
        self.trafficflag = float(99)

    # def update_target_graph(self, from_scope, to_scope):
    #     from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    #     to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    #
    #     op_holder = []
    #     for from_var, to_var in zip(from_vars, to_vars):
    #         op_holder.append(to_var.assign(from_var))
    #     return op_holder

    # Discounting function used to calculate discounted returns.
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    def train(self, espisode_buff, sess, value_est):
        espisode_buff = np.array(espisode_buff)
        input_p = espisode_buff[:self.batch_size, 0]
        input_v = espisode_buff[:self.batch_size, 1]
        actions = espisode_buff[:self.batch_size, 2]
        rewards = espisode_buff[:, 3]
        values = espisode_buff[:, 4]

        self.rewards_plus = np.asarray(rewards.tolist() + [value_est])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]
        discounted_rewards = np.append(discounted_rewards, 0)  # --
        discounted_rewards_batch = discounted_rewards[:self.batch_size] - (self.gamma ** self.batch_size) * discounted_rewards[self.batch_size:]  # --
        self.value_plus = np.asarray(values.tolist() + [value_est])
        # advantages = rewards + self.gamma * self.value_plus[1:] - self.value_plus[:-1]
        # advantages = self.discount(advantages)
        # advantages = np.append(advantages, 0) # --
        # advantages = advantages[:self.batch_size] - (self.gamma**self.batch_size)*advantages[self.batch_size:] # --'''
        advantages = discounted_rewards_batch - self.value_plus[:self.batch_size]

        # a filtering scheme, filter out 20% largest and smallest elements from 'discounted_rewards_batch'
        '''sorted_reward = np.argsort(discounted_rewards_batch)
        input_p = input_p[sorted_reward[10:-10]]
        input_v = input_v[sorted_reward[10:-10]]
        discounted_rewards_batch = discounted_rewards_batch[sorted_reward[10:-10]]
        actions = actions[sorted_reward[10:-10]]
        advantages = advantages[sorted_reward[10:-10]]'''

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_network.target_v: discounted_rewards_batch,  # _batch,
                     self.local_network.Input_p: np.vstack(input_p),
                     self.local_network.Input_v: np.vstack(input_v),
                     self.local_network.actions: actions,
                     self.local_network.advantages: advantages}
        sum_value_losss, sum_policy_loss, sum_entropy, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, _, _, regu_loss_policy, regu_loss_value = sess.run(
            [self.local_network.loss_value,
             self.local_network.loss_policy,
             self.local_network.entropy,
             self.local_network.grad_norms_policy,
             self.local_network.grad_norms_value,
             self.local_network.var_norms_policy,
             self.local_network.var_norms_value,
             self.local_network.apply_grads_policy,
             self.local_network.apply_grads_value,
             self.local_network.regu_loss_policy,
             self.local_network.regu_loss_value],
             feed_dict=feed_dict)
        return sum_value_losss / self.batch_size, sum_policy_loss / self.batch_size, sum_entropy / self.batch_size, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, regu_loss_policy / self.batch_size, regu_loss_value / self.batch_size

    def getInputFeature(self):
        ''' state (network matrix) reading'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/NetworkMatrix.txt", "r")
        line = f.readline()
        d = []
        while line:
            a = line.split(',')
            for x in a:
                d.append(x)
            line = f.readline()
        self.netflag = float(d[0])
        inputfeature = np.zeros((16, 16), dtype=float)  # empty，16 row 16 column
        for row in range(16):
            for col in range(16):
                inputfeature[row][col] = d[row * 16 + col + 1]
        f.close()
        return inputfeature

    def getInputFeature2(self):
        ''' state (traffic matrix) reading'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/TrafficMatrix.txt", "r")
        line = f.readline()
        while len(line) == 0:
            line = f.readline()
        d = []
        while line:
            a = line.split(',')
            for x in a:
                d.append(x)
            line = f.readline()
        inputfeature = np.zeros((16, 16), dtype=float)  # empty，16 row 16 column
        for row in range(16):
            for col in range(16):
                inputfeature[row][col] = d[row * 16 + col + 1]
        f.close()
        return inputfeature


    def getInputFeatureFlag(self):
        ''' state (network matrix) reading'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/NetworkMatrix.txt", "r")
        line = f.readline()
        d = []
        while line:
            a = line.split(',')
            for x in a:
                d.append(x)
            line = f.readline()
        self.netflag = float(d[0])
        f.close()
        return self.netflag

    def getInputFeatureFlag2(self):
        ''' state (trafffic matrix) reading'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/TrafficMatrix.txt", "r")
        line = f.readline()
        while len(line) == 0:
            line = f.readline()
        d = []
        while line:
            a = line.split(',')
            for x in a:
                d.append(x)
            line = f.readline()
        self.trafficflag = float(d[0])
        f.close()
        return self.trafficflag

    # def getReward(self):
    #     '''get reward from Delay.txt file'''
    #     f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/Delay.txt", "r")
    #     lines = f.readlines()
    #     d = []
    #     for v in lines:
    #         if len(v):
    #             a = v.split(',')
    #             for x in a:
    #                 d.append(x)
    #     f.close()
    #     if len(d):
    #         # if not isinstance(d[len(d)-1], str):
    #         #     r_t = float(d[len(d) - 1])  # get reward (mean E2E delay)
    #         if d[len(d) - 1] == 'nan' or d[len(d) - 1] == 'NAN':
    #             r_t = 0
    #         else:
    #             r_t = float(d[len(d) - 1])  # get reward (mean E2E delay)
    #     else:
    #         r_t = 0
    #     return r_t, len(lines)
    def getDelay(self):
        '''get delay from Delay.txt file'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/Delay.txt", "r")
        lines = f.readlines()
        d = []
        for v in lines:
            if len(v):
                a = v.split(',')
                for x in a:
                    d.append(x)
        f.close()
        if len(d):
            # if not isinstance(d[len(d)-1], str):
            #     r_t = float(d[len(d) - 1])  # get reward (mean E2E delay)
            if d[len(d) - 1] == 'nan' or d[len(d) - 1] == 'NAN':
                r_t = 0
            else:
                r_t = float(d[len(d) - 1])  # get reward (mean E2E delay)
        else:
            r_t = 0
        return r_t, len(lines)

    def getThroughput(self):
        '''get Throughput from Throughput.txt file'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/Throughput.txt", "r")
        lines = f.readlines()
        d = []
        for v in lines:
            if len(v):
                a = v.split(',')
                for x in a:
                    d.append(x)
        f.close()
        if len(d):
            if d[len(d) - 1] == 'nan' or d[len(d) - 1] == 'NAN':
                r_t = 0
            else:
                r_t = float(d[len(d) - 1])  # get Throughput
        else:
            r_t = 0
        return r_t, len(lines)

    def getPktLoss(self):
        '''get PktLoss from DropNum.txt file'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/DropNum.txt", "r")
        lines = f.readlines()
        d = []
        for v in lines:
            if len(v):
                a = v.split(',')
                for x in a:
                    d.append(x)
        f.close()
        if len(d):
            if d[len(d) - 1] == 'nan' or d[len(d) - 1] == 'NAN':
                r_t = 0
            else:
                r_t = float(d[len(d) - 1])  # get DropNum
        else:
            r_t = 0
        return r_t, len(lines)

    def writeFlag(self, string):
        '''NetworkMatrix.txt flag change writing'''
        f = open("E:/omnetpp-5.1.1-src-windows/omnetpp-5.1.1/Simu1.2Test2/NetworkMatrix.txt", "r+")
        lines = f.readlines()
        f.seek(0, 0)
        line_new = lines[0].replace(lines[0], string)
        f.write(line_new + '\n')
        for i in range(1, 17):
            f.write(lines[i])
        f.close()

    def minDelay(self, sess): # , coord, saver

        # time_to = 0
        # req_id = 0
        # episode_count = sess.run(self.global_episodes)    #???
        total_steps = 0
        episode_buffer = []


        action_onehot = [x for x in range(self.n_actions)]

        # update local dnn with the global one
        # sess.run(self.update_local_ops)

        print('Starting ' + self.name)
        with sess.as_default(), sess.graph.as_default():
            while 1:
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                actionss = []
                episode_buffer = []
                aver_delay = []

                # begin an episode
                while episode_step_count < self.episode_size:

                    '''Get Input_feature from NetworkMatrix.txt & TrafficMatrix.txt'''
                    Input_feature1 = self.getInputFeature()
                    Input_feature2 = self.getInputFeature2()
                    Input_feature = np.append(Input_feature1, Input_feature2)
                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p, self.y_dim_p, self.z_dim_p))
                    '''Input_feature = self.getInputFeatureFlag()
                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))'''
                    self.trafficflag = self.getInputFeatureFlag2()

                    # Take an action using probabilities from policy network output.
                    prob_dist, value, entro = sess.run(
                        [self.local_network.policy, self.local_network.value, self.local_network.entropy],
                        feed_dict={self.local_network.Input_p: Input_feature,
                                   self.local_network.Input_v: Input_feature})
                    pp = prob_dist[0]
                    assert not np.isnan(entro)
                    if self.name == 'agent_0':
                        print(pp, '--')
                    action_id = np.random.choice(action_onehot, p=pp) #+ 2 #2,3,4,5
                    # explore probability
                    e = 0.03
                    r = random.random()
                    if r <= e:
                        action_id = np.random.choice(action_onehot, p=[0.25,0.25,0.25,0.25])
                    print('action_id: ' + str(action_id), ' --')

                    if self.netflag == action_id + 2:
                        self.netflag = (action_id + 2)*10 + (action_id + 2)
                    else:
                        self.netflag = action_id + 2
                    actionss.append(action_id)
                    print('Take an action, action is ' + str(self.netflag) + '...')

                    # apply the selected action
                    '''NetworkMatrix.txt flag change writing'''
                    self.writeFlag(str(self.netflag))
                    # print('Writing flag ' + str(self.netflag) + ' to NetworkMatrix.txt...')

                    '''waiting for reward calculation'''
                    # rwd, numline = self.getReward()
                    # while numline == self.numline: # while rwd == self.delayflag:
                    #     rwd, numline = self.getReward()
                    #     # sleep(0.001)
                    # self.numline = numline # self.delayflag = rwd
                    # print('len(lines) changes to '+str(self.numline)+', Get a new Delay: ' + str(rwd) + '...')

                    dly, numline = self.getDelay()
                    while numline == self.numline:
                        dly, numline = self.getDelay()
                        # sleep(0.001)
                    self.numline = numline
                    print('len(lines) changes to '+str(self.numline)+', Get a new Delay: ' + str(dly))

                    # thrp, numline2 = self.getThroughput()
                    # while numline2 == self.numline2:
                    #     thrp, numline2 = self.getThroughput()
                    # self.numline2 = numline2
                    # print('len(lines) changes to '+str(self.numline2)+', Get a new Throughput: ' + str(thrp))
                    #
                    # drp, numline3 = self.getPktLoss()
                    # while numline3 == self.numline3:
                    #     drp, numline3 = self.getPktLoss()
                    # self.numline3 = numline3
                    # print('len(lines) changes to '+str(self.numline3)+', Get a new DropNum: ' + str(drp))

                    '''get reward from txt file'''
                    assert dly >= 0
                    # assert thrp >= 0
                    # assert drp >= 0
                    if dly != 0:
                        r_t = 0.0001/dly
                        # r_t = 10*thrp/((dly*100000)*((drp+1)*1024*8/0.05*e-9))
                    #
                    # if self.trafficflag == 0 :
                    #     if rwd != 0:
                    #         r_t = (1/rwd/50000)**4 # get reward (mean E2E delay--reward)
                    # elif self.trafficflag == 4:
                    #     if rwd != 0:
                    #         r_t = (1/(rwd-6e-6)/50000)**4 # get reward (mean E2E delay--reward)
                    # elif self.trafficflag == 6:
                    #     if rwd != 0:
                    #         r_t = (1/(rwd-3e-6)/50000)**4 # get reward (mean E2E delay--reward)
                    # else: # self.trafficflag == 5:
                    #     if rwd != 0:
                    #         r_t = (1/(rwd+3e-6)/50000)**4 # get reward (mean E2E delay--reward)
                    # r_t = round(r_t)
                    # if r_t > 10:
                    #     r_t = 10

                    print("self.trafficflag = " + str(self.trafficflag) + ', The new reward is: ' + str(r_t) + '...')
                    # store reward
                    fp = open('reward.dat', 'a')
                    fp.write(str(r_t) + '\n')
                    fp.close()
                    aver_delay.append(r_t)
                    #
                    if len(aver_delay) == 32:
                        ad = sum(aver_delay)/32
                        fp = open('reward32.dat', 'a')
                        fp.write(str(ad) + '\n')
                        fp.close()
                        aver_delay.clear()

                    episode_reward += r_t
                    total_steps += 1
                    episode_step_count += 1

                    # if episode_count < (3000 / self.episode_size):  # for warm-up
                    #     continue

                    # store experience
                    episode_buffer.append([Input_feature, Input_feature, action_id, r_t, value[0, 0]])
                    episode_values.append(value[0, 0])

                    if len(episode_buffer) == 2 * self.batch_size - 1:
                        mean_value_losss, mean_policy_loss, mean_entropy, grad_norms_policy, grad_norms_value, var_norms_policy, \
                        var_norms_value, regu_loss_policy, regu_loss_value = self.train(episode_buffer, sess, 0.0)
                        del (episode_buffer[:self.batch_size])
                    #    sess.run(self.update_local_ops)  # if we want to synchronize local with global every a training is performed

                # end of an episode
                # episode_count += 1

                # sess.run(self.update_local_ops) # if we want to synchronize local with global every episode is finished

                # if episode_count <= (3000 / self.episode_size):  # for warm-up
                #    continue


                if self.name == 'agent_0':
                    # print('Blocking Probability = ', bp)
                    print('Action Distribution', actionss.count(0) / len(actionss))
                    # store blocking probability
                    # fp = open('BP.dat', 'a')
                    # fp.write('%f\n' % bp)
                    # fp.close()
                    # store value prediction
                    fp = open('value.dat', 'a')
                    fp.write('%f\n' % np.mean(episode_values))
                    fp.close()
                    # store value loss
                    fp = open('value_loss.dat', 'a')
                    fp.write('%f\n' % float(mean_value_losss))
                    fp.close()
                    # store policy loss
                    fp = open('policy_loss.dat', 'a')
                    fp.write('%f\n' % float(mean_policy_loss))
                    fp.close()
                    # store entroy
                    fp = open('entropy.dat', 'a')
                    fp.write('%f\n' % float(mean_entropy))
                    fp.close()

                # self.episode_blocking.append(bp)
                self.episode_rewards.append(episode_reward)
                self.episode_mean_values.append(np.mean(episode_values))

                # Periodically save model parameters, and summary statistics.
                # sample_step = int(1000 / self.episode_size)
                # if episode_count % sample_step == 0 and episode_count != 0:
                #         '''if episode_count % (100*sample_step) == 0 and self.name == 'agent_0':
                #             saver.save(sess, self.model_path+'/model.cptk')
                #             print ("Model Saved")'''
                #
                #         '''if self.name == 'agent_0':
                #             mean_reward = np.mean(self.episode_rewards[-sample_step:])
                #             mean_value = np.mean(self.episode_mean_values[-sample_step:])
                #             mean_blocking = np.mean(self.episode_blocking[-sample_step:])
                #             summary = tf.Summary()
                #             summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                #             summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                #             summary.value.add(tag='Perf/Blocking', simple_value=float(mean_blocking))
                #             summary.value.add(tag='Losses/Value Loss', simple_value=float(mean_value_losss))
                #             summary.value.add(tag='Losses/Policy Loss', simple_value=float(mean_policy_loss))
                #             summary.value.add(tag='Losses/Entropy', simple_value=float(mean_entropy))
                #             summary.value.add(tag='Losses/Grad Norm Policy', simple_value=float(grad_norms_policy))
                #             summary.value.add(tag='Losses/Grad Norm Value', simple_value=float(grad_norms_value))
                #             summary.value.add(tag='Losses/Var Norm Policy', simple_value=float(var_norms_policy))
                #             summary.value.add(tag='Losses/Var Norm Value', simple_value=float(var_norms_value))
                #             summary.value.add(tag='Losses/Regu Loss Policy', simple_value=float(regu_loss_policy))
                #             summary.value.add(tag='Losses/Regu Loss Value', simple_value=float(regu_loss_value))
                #             self.summary_writer.add_summary(summary, episode_count)
                #
                #             self.summary_writer.flush()'''
                # if self.name == 'agent_0':
                #        sess.run(self.increment)


if __name__ == "__main__":

    n_actions = 4
    NODE_NUM = 16
    # x_dim_p = 16*16 # 1
    # x_dim_v = 16*16 # 1
    num_layers = 2 #5
    layer_size = 16 #128
    regu_scalar = 1e-4

    filter_size = 2
    num_filters = 256
    num_residuals_policy = 4
    regu_scalar = 1e-3
    num_filters_last_cov = 1

    gamma = 0.95 # penalty on future reward
    episode_size = 100 # 1000 # number of rewards in each episode
    batch_size = 50

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)#1e-5
        # trainer = tf.train.RMSPropOptimizer(learning_rate = 1e-5, decay = 0.99, epsilon = 0.0001)
        agent = NetRec_Agent(0, trainer, gamma, episode_size, batch_size, n_actions, NODE_NUM, filter_size, num_filters, num_residuals_policy, regu_scalar, num_filters_last_cov)

    with tf.Session() as sess:
        # coord = tf.train.Coordinator()
        # if load_model == True:
        #     ckpt = tf.train.get_checkpoint_state(model_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer())

        agent.minDelay(sess)
