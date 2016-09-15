#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
logs= sys.stderr

import mxnet as mx
import numpy as np
import time
import math
import logging

from collections import namedtuple

import data_iter

ReLUModel = namedtuple("ReLUModel", ['relu_exec', 'symbol',
                                     'data', 'label', 'param_blocks'])

class GreedyParser(object):
    """Builds a Chen & Manning style greedy neural net parser.

    Args:
        num_actions: int size of the set of parser actions.
        num_features: int list of dimensions of the feature vectors.
        num_feature_ids: int list of same length as num_features corresponding to the sizes of the input feature spaces.
        embedding_sizes: int list of same length as num_features of the desired embedding layer sizes.
    """
    def __init__(self,
                 num_actions,
                 num_features,
                 num_feature_ids,
                 embedding_sizes,
                 hidden_layer_sizes,
                 learning_rate=0.1,
                 max_grad_norm=5.0,
                 epoch=13,
                 optimizer='sgd',
                 decay_steps=4000):
        self._num_actions = num_actions
        self._num_features = num_features
        self._num_feature_ids = num_feature_ids
        self._embedding_sizes = embedding_sizes
        self._hidden_layer_sizes = hidden_layer_sizes
        self._learning_rate = learning_rate
        self._max_grad_norm = max_grad_norm
        self._optimizer = optimizer
        self._epoch = epoch
        self._hidden_layer_sizes = hidden_layer_sizes
        self._feature_size = len(embedding_sizes)
        self._decay_steps = decay_steps
        self._step = 0
        self._decay_rate = 0.96

    @property
    def embedding_size(self,):
        size = 0
        for i in range(self._feature_size):
            size += self._num_features[i] * self._embedding_sizes[i]
        return size

    def _AddParam(self, shape, initializer=None, return_average=False):
        pass

    def _AddEmbedding(self, num_features, vocab_size, embedding_size, name):
        data = mx.sym.Variable('feature_%s_data' % name)
        embed_weight = mx.sym.Variable('%s_embed_weight' % name)
        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                                  input_dim=vocab_size, output_dim=embedding_size)
        hidden = mx.sym.Reshape(hidden, target_shape=(0, num_features * embedding_size))
        return hidden

    def _BuildNetwork(self,):
        # Create embedding layer.
        embeddings = []
        for i in range(self._feature_size):
            embeddings.append(self._AddEmbedding(self._num_features[i],
                self._num_feature_ids[i], self._embedding_sizes[i], i))

        last_layer = mx.sym.Concat(*embeddings, dim=1)
        last_layer_size = self.embedding_size

        # Create ReLU layers.
        for i, hidden_layer_size in enumerate(self._hidden_layer_sizes):
            i2h_weight = mx.sym.Variable('t_%d_i2h_weight' % i)
            i2h_bias = mx.sym.Variable('t_%d_i2h_bias' % i)
            last_layer = mx.sym.FullyConnected(data=last_layer,
                    weight= i2h_weight,
                    bias = i2h_bias,
                    num_hidden=last_layer_size)
            last_layer = mx.sym.Activation(data=last_layer, act_type='relu')
            last_layer_size = hidden_layer_size

        # Create Softmax layer.
        label = mx.sym.Variable('label')
        softmax_weight = mx.sym.Variable('softmax_weight')
        softmax_bias = mx.sym.Variable('softmax_bias')
        fc = mx.sym.FullyConnected(data=last_layer, weight=softmax_weight,
                                   bias=softmax_bias, num_hidden=self._num_actions)
        sm = mx.sym.SoftmaxOutput(data=fc, label=label)
        return sm

    def _IsParameter(self, name):
        return name.endswith('weight') or name.endswith('bias')

    def SetupModel(self, ctx, batch_size, initializer=mx.initializer.Uniform(0.2)):
        self._batch_size = batch_size

        relu_model = self._BuildNetwork()
        arg_names = relu_model.list_arguments()

        # Setup input data shape
        input_shapes = {}
        for i in range(self._feature_size):
            input_shapes['feature_%d_data' % i] = (batch_size, self._num_features[i])

        # Infer shape
        arg_shape, out_shape, aux_shape = relu_model.infer_shape(**input_shapes)
        arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
        arg_grads = {}
        for shape, name in zip(arg_shape, arg_names):
            if self._IsParameter(name):
                arg_grads[name] = mx.nd.zeros(shape, ctx)

        relu_exec = relu_model.bind(ctx=ctx, args=arg_arrays,
                                    args_grad=arg_grads, grad_req='add')

        param_blocks = []
        arg_dict = dict(zip(arg_names, relu_exec.arg_arrays))
        for i, name in enumerate(arg_names):
            if self._IsParameter(name):
                initializer(name, arg_dict[name])
                param_blocks.append( (i, arg_dict[name], arg_grads[name], name) )

        out_dict = dict(zip(relu_model.list_outputs(), relu_exec.outputs))

        data = [relu_exec.arg_dict['feature_%d_data' % i] for i in range(self._feature_size)]
        label = relu_exec.arg_dict['label']

        self._relu_model = ReLUModel(relu_exec=relu_exec, symbol=relu_model,
                                     data=data, label=label, param_blocks=param_blocks)

    def TrainModel(self, X_train, y_train):
        m = self._relu_model
        # Create optimizer
        opt = mx.optimizer.create(self._optimizer)
        opt.lr = self._learning_rate
        opt.wd = 0.0001
        opt.momentum = 0.9
        updater = mx.optimizer.get_updater(opt)

        print >> logs, "start training..."
        for iteration in range(self._epoch):
            tic = time.time()
            num_correct = 0
            num_total = 0
            # TODO:: use dataIter instead (for padding).
            for begin in range(0, X_train.shape[0], self._batch_size):
                batchX = X_train[begin:begin+self._batch_size]
                batchY = y_train[begin:begin+self._batch_size]
                if batchX.shape[0] != self._batch_size:
                    continue

                # decay learning rate.
                if self._step > self._decay_steps and self._step % self._decay_steps == 0:
                    self._learning_rate *= self._decay_rate ** (int(self._step /
                                                                    self._decay_steps))
                    opt.lr = self._learning_rate
                    print >> logs, 'decay learning rate, now lr is [%.6f], global step [%d]' % (opt.lr, self._step)
                
                # accumlating step.
                self._step += 1

                start = 0
                for i in range(self._feature_size):
                    m.data[i][:] = batchX[:,start:start+self._num_features[i]]
                    start = start + self._num_features[i]
                m.label[:] = batchY

                m.relu_exec.forward(is_train=True)
                m.relu_exec.backward()

                num_correct += sum(batchY == np.argmax(m.relu_exec.outputs[0].asnumpy(), axis=1))
                num_total += len(batchY)

                # update weights
                norm = 0
                for idx, weight, grad, name in m.param_blocks:
                    grad /= self._batch_size
                    l2_norm = mx.nd.norm(grad).asscalar()
                    norm += l2_norm * l2_norm
                norm = math.sqrt(norm)
                for idx, weight, grad, name in m.param_blocks:
                    if norm > self._max_grad_norm:
                        grad *= self._max_grad_norm / norm

                    updater(idx, grad, weight)

                    # Reset gradient to zero
                    grad[:] = 0.0


            # End of training loop
            toc = time.time()
            train_time = toc - tic
            train_acc = num_correct * 100 / float(num_total)

            print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)

            if iteration == 9:
                prefix = 'greedy'
                self._relu_model.symbol.save('%s-symbol.json' % prefix)
                save_dict = {('arg:%s' % k) : v for k, v in self._relu_model.relu_exec.arg_dict.items()
                        if self._IsParameter(k) }
                save_dict.update({('aux:%s' % k) : v for k, v in self._relu_model.relu_exec.aux_dict.items() })
                param_name = '%s-%04d.params' % (prefix, iteration)
                mx.nd.save(param_name, save_dict)
                print >> logs, 'Saved model %s' % param_name


    def plot_network(self):
        symbol = self._relu_model.symbol
        dot = mx.viz.plot_network(symbol, title='greedy_parser')
        dot.render(filename='greedy_parser')


def main():
    num_actions = 37
    num_features = [20, 20, 12]
    num_feature_ids = [34346, 34, 21]
    embedding_sizes = [64, 32, 32]
    hidden_layer_sizes = [200, 200]
    batch_size = 32
    xdata, ydata = data_iter.read_data(sys.argv[1])
    parser = GreedyParser(num_actions, num_features, num_feature_ids, embedding_sizes, hidden_layer_sizes)
    parser.SetupModel(mx.gpu(0), batch_size=batch_size)
    parser.TrainModel(xdata, ydata)
    # parser.plot_network()

if __name__ == '__main__':
    main()
