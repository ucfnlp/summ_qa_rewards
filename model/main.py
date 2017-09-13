import os, sys, gzip
import time
import math
import json
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T

from nn.optimization import create_optimization_updates
from nn.advanced import sigmoid, linear, RCNN
from nn.initialization import  get_activation_by_name
from nn.basic import EmbeddingLayer, Layer, LSTM, apply_dropout
from util import say
import myio
from nn.extended_layers import ExtRCNN, ExtLSTM, ZLayer
import summarization_args


def build_model():
    args = summarization_args.get_args()


    embedding_layer = embedding_layer = myio.create_embedding_layer(
                        args.embedding
                    )

    padding_id = embedding_layer.vocab_map["<padding>"]

    dropout = theano.shared(
        np.float64(args.dropout).astype(theano.config.floatX)
    )

    # len*batch
    x = T.imatrix()

    n_d = args.embedding_dim
    n_e = embedding_layer.n_d
    activation = get_activation_by_name(args.activation)

    layers  = []

    for i in xrange(2):
        l = LSTM(
            n_in=n_e,
            n_out=n_d,
            activation=activation
        )

        layers.append(l)

    # len * batch
    masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

    # (len*batch)*n_e
    embs = embedding_layer.forward(x.ravel())
    # len*batch*n_e
    embs = embs.reshape((x.shape[0], x.shape[1], n_e))
    embs = apply_dropout(embs, dropout)

    flipped_embs = embs[::-1]

    # len*bacth*n_d
    h1 = layers[0].forward_all(embs)
    h2 = layers[1].forward_all(flipped_embs)
    h_final = T.concatenate([h1, h2[::-1]], axis=2)
    h_final = apply_dropout(h_final, dropout)
    size = n_d * 2


if __name__ == '__main__':
    build_model()