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

    embedding_layer = myio.create_embedding_layer(args.embedding)

    padding_id = embedding_layer.vocab_map["<padding>"]

    dropout = theano.shared(
        np.float64(args.dropout).astype(theano.config.floatX)
    )

    x = T.imatrix()

    n_d = args.embedding_dim
    n_e = embedding_layer.n_d
    activation = get_activation_by_name(args.activation)

    layers = []

    for i in xrange(2):
        l = LSTM(
            n_in=n_e,
            n_out=n_d,
            activation=activation
        )

        layers.append(l)

    # len * batch
    masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

    embs = embedding_layer.forward(x.ravel())

    embs = embs.reshape((x.shape[0], x.shape[1], n_e))
    embs = apply_dropout(embs, dropout)

    flipped_embs = embs[::-1]

    h1 = layers[0].forward_all(embs)
    h2 = layers[1].forward_all(flipped_embs)

    h_final_word = T.concatenate([h1, h2[::-1]], axis=2)
    h_final_word = apply_dropout(h_final_word, dropout)

    h_final_sent = T.concatenate([h1[args.sentence_length-1::args.sentence_length],
                                  h2[::args.sentence_length]], axis=2)
    # Todo: apply dropout here
    # h_final_sent = apply_dropout(h_final_word, dropout)

    size = n_d * 2

    # Word Level
    output_layer_word = ZLayer(
                n_in = size,
                n_hidden = args.sentence_length,
                activation = activation
            )

    z_pred_word, sample_updates = output_layer_word.sample_all(h_final_word)

    z_pred_word = theano.gradient.disconnected_grad(z_pred_word)

    # Sentence Level
    output_layer_sentence = ZLayer(
        n_in=size,
        n_hidden=args.max_sentences,
        activation=activation
    )

    z_pred_sent, sample_updates = output_layer_sentence.sample_all(h_final_sent)

    z_pred_sent = theano.gradient.disconnected_grad(z_pred_sent)




if __name__ == '__main__':
    build_model()