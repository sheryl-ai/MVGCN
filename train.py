# %load_ext autoreload
# %autoreload 2
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.insert(0, '..')
import models, graph, coarsening, utils
# from utils import model_perf

import tensorflow as tf
import numpy as np
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import scipy.sparse as sp
import pickle as pkl

# from tensorflow.examples.tutorials.mnist import input_data

# %matplotlib inline
flags = tf.app.flags
FLAGS = flags.FLAGS

# neural network setting
# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

# Directories.
# flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')

results_auc = dict()
results = list()

class model_perf(object):

    def __init__(self):
        self.names, self.params = set(), {}
        self.fit_auc, self.fit_losses, self.fit_time = {}, {}, {}
        self.train_auc, self.train_loss = {}, {}
        self.test_auc, self.test_loss = {}, {}
        self.train_represent = {}


    def test(self, model, name, params, data, train_pairs, train_labels, val_data, val_labels, test_pairs, test_labels):
        self.params[name] = params
        self.fit_auc[name], self.fit_losses[name], self.fit_time[name] = \
                model.fit(data, train_pairs, train_labels, val_data, val_labels)
        # string, self.train_auc[name], self.train_loss[name], _ = \
        #         model.evaluate(train_data, train_labels)
        # print('train {}'.format(string))
        del val_data, val_labels
        n, v, m, f = data.shape
        if params['method'] == 'gcn' or params['method'] == '2gcn':
            test_data = np.zeros([test_pairs.shape[0], v, m, f, 2])
            test_data[:,:,:,:,0] = data[test_pairs[:,0], :, :, :]
            test_data[:,:,:,:,1] = data[test_pairs[:,1], :, :, :]
        elif params['method'] == 'fnn' or params['method'] == '2fnn':
            new_data = np.zeros([n, v, m*f])
            for i in range(n):
                for j in range(v):
                    new_data[i, j, :] = data[i, j, :, :].flatten()
            test_data = np.zeros([test_pairs.shape[0], v, m*f, 2])
            test_data[:,:,:,0] = new_data[test_pairs[:,0], :, :]
            test_data[:,:,:,1] = new_data[test_pairs[:,1], :, :]
        print (test_data.shape)
        string, self.test_auc[name], self.test_loss[name], _, = \
                model.evaluate(test_data, test_labels)
        print('test  {}'.format(string))

        # f = open('test.roi.gcn.pkl', 'wb')
        # pkl.dump(test_represent, f, -1)
        # f.close()
        # f = open('test.roi.pairs.gcn.pkl', 'wb')
        # pkl.dump(test_pairs, f, -1)
        # f.close()
        # self.save_represent(model, data, train_pairs, train_labels, test_pairs, test_labels, params)
        self.names.add(name)

    def save_represent_fnn(self, model, data, train_pairs, train_labels, test_pairs, test_labels, params):

        n, v, m, f = data.shape
        new_data = np.zeros([n, v, m*f])
        for i in range(n):
            for j in range(v):
                new_data[i, j, :] = data[i, j, :, :].flatten()

        n_train = train_pairs.shape[0]
        num = int(np.ceil(n_train/10))
        represent = np.zeros([n_train, 84], dtype='float32')
        for i in range(10): # training data, split into 10 sets
            if (i+1)*num <= n_train:
                tmp_pairs = train_pairs[i*num:(i+1)*num,:]
                train_x = np.zeros([tmp_pairs.shape[0], v, m*f, 2])
                train_x[:,:,:,0] = new_data[tmp_pairs[:,0], :, :]
                train_x[:,:,:,1] = new_data[tmp_pairs[:,1], :, :]
                train_y = train_labels[i*num:(i+1)*num]
                print (train_x.shape)
                represent[i*num:(i+1)*num,:] = model.get_represent(train_x, train_y)
            else:
                tmp_pairs = train_pairs[i*num:,:]
                train_x = np.zeros([tmp_pairs.shape[0], v, m*f, 2])
                train_x[:,:,:,0] = new_data[tmp_pairs[:,0], :, :]
                train_x[:,:,:,1] = new_data[tmp_pairs[:,1], :, :]
                train_y = train_labels[i*num:]
                print (train_x.shape)
                represent[i*num:,:] = model.get_represent(train_x, train_y)
        f = open('train.roi.eu.gcn.pkl', 'wb')
        pkl.dump(represent, f, -1)
        f.close()
        f = open('train.roi.pairs.eu.gcn.pkl', 'wb')
        pkl.dump(train_pairs, f, -1)
        f.close()


    def save_represent(self, model, data, train_pairs, train_labels, test_pairs, test_labels, params):

        n, v, m, f = data.shape
        n_train = train_pairs.shape[0]
        num = int(np.ceil(n_train/20))
        represent = np.zeros([n_train, 84], dtype='float32')
        for i in range(20): # training data, split into 10 sets
            if (i+1)*num <= n_train:
                tmp_pairs = train_pairs[i*num:(i+1)*num,:]
                train_x = np.zeros([tmp_pairs.shape[0], v, m, f, 2])
                train_x[:,:,:,:,0] = data[tmp_pairs[:,0], :, :, :]
                train_x[:,:,:,:,1] = data[tmp_pairs[:,1], :, :, :]
                train_y = train_labels[i*num:(i+1)*num]
                print (train_x.shape)
                represent[i*num:(i+1)*num,:] = model.get_represent(train_x, train_y)
            else:
                tmp_pairs = train_pairs[i*num:,:]
                train_x = np.zeros([tmp_pairs.shape[0], v, m, f, 2])
                train_x[:,:,:,:,0] = data[tmp_pairs[:,0], :, :, :]
                train_x[:,:,:,:,1] = data[tmp_pairs[:,1], :, :, :]
                train_y = train_labels[i*num:]
                print (train_x.shape)
                represent[i*num:,:] = model.get_represent(train_x, train_y)
        f = open('train.roi.gcn.pkl', 'wb')
        pkl.dump(represent, f, -1)
        f.close()
        f = open('train.roi.pairs.gcn.pkl', 'wb')
        pkl.dump(train_pairs, f, -1)
        f.close()


    def save(self, data_type):
        results = list()
        for name in sorted(self.names):
            results.append([name, self.test_accuracy[name], self.train_accuracy[name],
            self.test_f1[name], self.train_f1[name], self.test_loss[name],
            self.train_loss[name], self.fit_time[name]*1000])

        if os.path.exists(data_type + '_results.csv'):
            old = pd.read_csv(data_type + '_results.csv', header=None)
            new = pd.DataFrame(data=results)
            r = pd.concat([old, new], ignore_index=True)
            r.to_csv(data_type + '_results.csv', index=False, header=['method', 'test_acc',
            'train_acc', 'test_f1', 'train_f1', 'test_loss', 'train_loss', 'fit_time'])
        else:
            r = pd.DataFrame(data=results)
            r.to_csv(data_type + '_results.csv', index=False, header=['method', 'test_acc',
            'train_acc', 'test_f1', 'train_f1', 'test_loss', 'train_loss', 'fit_time'])


    def fin_result(self, data_type, i_fold=None):
        for name in sorted(self.names):
            if name not in results_auc:
                results_auc[name] = 0
            results_auc[name] += self.test_auc[name]
            results.append([i_fold, self.test_auc[name]])
        if i_fold == 4:
            for name in sorted(self.names):
                results_auc[name] /= 5
                print('{:5.2f}  {}'.format(
                    results_auc[name], name))
            results.append([name, results_auc[name]])
            r = pd.DataFrame(data=results)
            r.to_csv('../../../data/output/' + data_type + '_fin_results', index=False, header=['method', 'test_auc'])


    def show(self, fontsize=None):
        if fontsize:
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)         # controls default text sizes
            plt.rc('axes', titlesize=fontsize)    # fontsize of the axes title
            plt.rc('axes', labelsize=fontsize)    # fontsize of the x any y labels
            plt.rc('xtick', labelsize=fontsize)   # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)   # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)   # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('  auc      loss        time [ms]  name')
        print('test  train   test  train   test     train')
        for name in sorted(self.names):
            print('{:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                    self.test_auc[name], self.train_auc[name],
                    self.test_loss[name], self.train_loss[name], self.fit_time[name]*1000, name))


def preprocess_features(features, scale=1):
    """ Row-normalized by divide maximum of the row"""
    rowmax = np.max(features, axis=1).reshape(features.shape[0], 1)
    features = np.int_(scale*np.divide(features, rowmax))
    return features


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
    return A


def get_feed_data(data, subj, pairs, labels, method='gcn'):
    train_pairs, val_pairs, test_pairs = pairs
    train_labels, val_labels, test_labels = labels
    n, v, m, f = data.shape
    if v == 6:
        print (val_labels.shape)
        n_val_pairs = 10000
        sidx = np.random.permutation(val_labels.shape[0])
        val_pairs =  np.array([val_pairs[s, :] for s in sidx[:n_val_pairs]])
        val_labels = np.array([val_labels[s] for s in sidx[:n_val_pairs]])
        # test_pairs = np.array([test_pairs[s, :] for s in sidx[n_val_pairs:]])
        # test_labels = np.array([test_labels[s] for s in sidx[n_val_pairs:]])
    # f = 1 # whether f can be deleted
    if method == 'gcn' or method == '2gcn':
        val_x = np.zeros([val_pairs.shape[0], v, m, f, 2])
        val_x[:,:,:,:,0] = data[val_pairs[:,0], :, :, :]
        val_x[:,:,:,:,1] = data[val_pairs[:,1], :, :, :]
    elif method == 'fnn' or method == '2fnn':
        new_data = np.zeros([n, v, m*f])
        for i in range(n):
            for j in range(v):
                new_data[i, j, :] = data[i, j, :, :].flatten()
        val_x = np.zeros([val_pairs.shape[0], v, m*f, 2])
        val_x[:,:,:,0] = new_data[val_pairs[:,0], :, :]
        val_x[:,:,:,1] = new_data[val_pairs[:,1], :, :]

    train_y = train_labels
    val_y = val_labels
    test_y = test_labels
    del subj
    del train_labels, val_labels, test_labels
    del val_pairs
    return train_pairs, train_y, val_x, val_y, test_pairs, test_y


def train(method, view_com, n_views, k, m, n_epoch, batch_size, pairs, labels, coords, subj, data, data_type, i_fold):
    str_params = view_com + '_k' + str(k) + '_m' + str(m) + '_'
    obj_params = 'softmax'
    print (str_params)

    print ('Construct ROI graphs...')
    t_start = time.process_time()
    # A = grid_graph(86, corners=False)
    # A = graph.replace_random_edges(A, 0)
    coo1, coo2, coo3 = coords.shape # coo2 is the roi dimension
    features = np.zeros([coo1*coo3, coo2])
    for i in range(coo3):
        features[coo1*i:coo1*(i+1), :] = coords[:, :, i]
    dist, idx = graph.distance_scipy_spatial(np.transpose(features), k=10, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)

    if method == '2gcn':
        graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]
        data = coarsening.perm_data1(data, perm, True)
    else:
        graphs = list()
        graphs.append(A)
        L = [graph.laplacian(A, normalized=True)]

    print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    # graph.plot_spectrum(L)
    del A

    print ('Set parameters...')
    mp = model_perf()
    # Architecture.
    common = {}
    common['dir_name']       = 'ppmi/'
    common['num_epochs']     = n_epoch
    common['batch_size']     = batch_size
    common['eval_frequency'] = 5 * common['num_epochs']
    common['patience']       = 5
    common['regularization'] = 5e-3
    common['dropout']        = 1
    common['learning_rate']  = 1e-2
    common['decay_rate']     = 0.95
    common['momentum']       = 0.9
    common['n_views']        = n_views
    common['view_com']       = view_com
    # common['brelu']          = 'b1relu'
    # common['pool']           = 'mpool1'

    print ('Get feed pairs and labels...')
    train_pairs, train_y, val_x, val_y, test_pairs, test_y = get_feed_data(data, subj, pairs, labels, method)
    C = max(train_y)+1
    common['decay_steps']    = train_pairs.shape[0] / (common['batch_size'] * 5)


    if method == 'fnn':
        str_params += 'siamese_'
        name = 'mvfnn'
        params = common.copy()
        params['method'] = 'fnn'
        params['fin']            = 1
        params['F']              = [m]
        params['K']              = [1]
        params['p']              = [1]
        params['M']              = [C]
        params['dir_name'] += name
        mp.test(models.siamese_fnn(L, **params), name, params,
        data, train_pairs, train_y, val_x, val_y, test_pairs, test_y)

    if method == '2fnn':
        str_params += 'siamese_layer2_'
        name = 'mvfnn2'
        params = common.copy()
        params['method'] = 'fnn'
        params['fin']            = 1
        params['F']              = [m]
        params['K']              = [1]
        params['p']              = [1]
        params['M']              = [64, C]
        params['dir_name'] += name
        mp.test(models.siamese_fnn(L, **params), name, params,
        data, train_pairs, train_y, val_x, val_y, test_pairs, test_y)

    if method == 'gcn':
        # str_params += 'b_max_eu_'
        name = 'mvgcn'
        params = common.copy()
        params['method'] = 'gcn'
        params['F']              = [m] # filters
        params['K']              = [k] # supports
        params['p']              = [1]
        params['M']              = [C]
        params['fin'] = val_x.shape[3]
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        params['brelu'] = 'b2relu'
        params['pool'] = 'apool1'
        mp.test(models.siamese_m_cgcnn(L, **params), name, params,
                        data, train_pairs, train_y, val_x, val_y, test_pairs, test_y)

    # Common hyper-parameters for LeNet5-like networks.
    if method == '2gcn':
        str_params += 'p4_fc64_'
        name = 'mvgcn2'
        params = common.copy()
        params['method'] = '2gcn'
        params['F']              = [m, 64] # filters
        params['K']              = [k, k] # supports
        params['p']              = [4, 4]
        params['M']              = [512, C]
        params['fin'] = val_x.shape[3]
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        params['brelu'] = 'b2relu'
        params['pool'] = 'apool1'
        mp.test(models.siamese_m_cgcnn(L, **params), name, params,
                        data, train_pairs, train_y, val_x, val_y, test_pairs, test_y)

    # mp.save(data_type)
    method_type = method + '_'
    mp.fin_result(method_type + data_type + str_params + obj_params, i_fold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str)
    parser.add_argument('data_type1', type=str)
    parser.add_argument('data_type2', type=str)
    parser.add_argument('data_type3', type=str)
    parser.add_argument('data_type4', type=str)
    parser.add_argument('data_type5', type=str)
    parser.add_argument('data_type6', type=str)
    parser.add_argument('view_com', type=str)
    parser.add_argument('kfold', type=str)
    parser.add_argument('K', type=int)
    parser.add_argument('M', type=int)
    parser.add_argument('n_epoch', type=int)
    parser.add_argument('batch_size', type=int)
    args = parser.parse_args()
    print ('---------------------------------------')
    data_type = [args.data_type1, args.data_type2, args.data_type3, args.data_type4, args.data_type5, args.data_type6]
    n_views = len(data_type)
    # See function train for all possible parameter and there definition.
    data, subj, coords, pairs, labels = utils.load_data(data_type=data_type, kfold=args.kfold)
    data_type = args.data_type1 + '+' + args.data_type2  + '+' + args.data_type3  + '+' + args.data_type4 + '+' + args.data_type5  + '+' + args.data_type6
    print (data.shape)
    if args.kfold == 'True':
        for l in range(5):
            if l >= 1:
                break
            print ("The %d fold ..." %(l+1))
            train(method=args.method,
                  view_com=args.view_com,
                  n_views=n_views,
                  k=args.K,
                  m=args.M,
                  n_epoch=args.n_epoch,
                  batch_size=args.batch_size,
                  pairs=pairs[l],
                  labels=labels[l],
                  coords=coords,
                  subj=subj,
                  data=data,
                  data_type=data_type,
                  i_fold=l)
    else:
        print ('fixed split')
        train(method=args.method,
                view_com=args.view_com,
                n_views=n_views,
                k=args.K,
                m=args.M,
                n_epoch=args.n_epoch,
                batch_size=args.batch_size,
                pairs=pairs,
                labels=labels,
                coords=coords,
                subj=subj,
                data=data,
                data_type=data_type)
