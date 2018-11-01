# coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

ll = ctypes.cdll.LoadLibrary
lib = ll("./init.so")


class Config(object):

    def __init__(self):
        self.L1_flag = True
        self.hidden_size = 50
        self.nbatches = 100
        self.entity = 0
        self.relation = 0
        self.trainTimes = 3000
        self.margin = 1.0


class TransEModel(object):

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size
        margin = config.margin

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[
                                                  entity_total, size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[
                                                  relation_total, size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
            
            #self.pos_r = tf.Print(self.pos_r, [self.pos_r, tf.shape(self.pos_r), "pos_r"])
            #pos_r_e = tf.Print(pos_r_e, [pos_r_e, tf.shape(pos_r_e), "pos_r_e"])
            #self.neg_r = tf.Print(self.neg_r, [self.neg_r, tf.shape(self.neg_r), "neg_r"])
            #neg_r_e = tf.Print(neg_r_e, [neg_r_e, tf.shape(neg_r_e), "neg_r_e"])
            
            M_pos_r = tf.get_variable("M_pos_r", [relation_total, size, size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            M_neg_r = tf.get_variable("M_neg_r", [relation_total, size, size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            M_pos_r_e = tf.nn.embedding_lookup(M_pos_r, self.pos_r)
            M_neg_r_e = tf.nn.embedding_lookup(M_neg_r, self.neg_r)
        
        def prod(var, M):
            new_var = tf.einsum('ai,aij->aj', var, M)
            return new_var
        
        pos_h_r = prod(pos_h_e, M_pos_r_e)
        pos_t_r = prod(pos_t_e, M_pos_r_e)
        
        neg_h_r = prod(neg_h_e, M_neg_r_e)
        neg_t_r = prod(neg_t_e, M_neg_r_e)

        pos = pos_h_r + pos_r_e - pos_t_r
        #pos = tf.reduce_sum(pos, axis=0)
        neg = neg_h_r + neg_r_e - neg_t_r
        #neg = tf.reduce_sum(neg, axis=0)
        
        energy = margin + tf.norm(pos, ord=2, axis=1)**2 - tf.norm(neg, ord=2, axis=1)**2
        energy = tf.maximum(energy, tf.constant(0.))
        # TO DO
        #energy = tf.Print(energy, [energy, tf.shape(energy), "energy"])

        with tf.name_scope("output"):
            # TO DO
            self.loss = tf.reduce_sum(energy)
            #self.loss = tf.Print(self.loss, [self.loss, tf.shape(self.loss), "loss"])


def main(_):
    lib.init()
    config = Config()
    config.relation = lib.getRelationTotal()
    config.entity = lib.getEntityTotal()
    config.batch_size = lib.getTripleTotal() / config.nbatches

    with tf.Graph().as_default():
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.15
        sess = tf.Session(config=config_gpu)
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                trainModel = TransEModel(config=config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(0.001)
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())

            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss = sess.run(
                    [train_op, global_step, trainModel.loss], feed_dict)
                return loss

            ph = np.zeros(config.batch_size, dtype=np.int32)
            pt = np.zeros(config.batch_size, dtype=np.int32)
            pr = np.zeros(config.batch_size, dtype=np.int32)
            nh = np.zeros(config.batch_size, dtype=np.int32)
            nt = np.zeros(config.batch_size, dtype=np.int32)
            nr = np.zeros(config.batch_size, dtype=np.int32)

            ph_addr = ph.__array_interface__['data'][0]
            pt_addr = pt.__array_interface__['data'][0]
            pr_addr = pr.__array_interface__['data'][0]
            nh_addr = nh.__array_interface__['data'][0]
            nt_addr = nt.__array_interface__['data'][0]
            nr_addr = nr.__array_interface__['data'][0]

            for times in range(config.trainTimes):
                res = 0.0
                for batch in range(config.nbatches):
                    lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr,
                                 nt_addr, nr_addr, config.batch_size)
                    res += train_step(ph, pt, pr, nh, nt, nr)
                    current_step = tf.train.global_step(sess, global_step)
                print times
                print res
            #saver.save(sess, 'model.vec')
            # save the embeddings
            f = open("entity2vec_R.txt", "w")
            enb = trainModel.ent_embeddings.eval()
            for i in enb:
                for j in i:
                    f.write("%f\t" % (j))
                f.write("\n")
            f.close()

            f = open("relation2vec_R.txt", "w")
            enb = trainModel.rel_embeddings.eval()
            for i in enb:
                for j in i:
                    f.write("%f\t" % (j))
                f.write("\n")
            f.close()

if __name__ == "__main__":
    tf.app.run()
