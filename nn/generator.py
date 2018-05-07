import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

from nn.basic import LSTM, apply_dropout
from nn.extended_layers import ZLayer
from nn.initialization import get_activation_by_name
from nn.advanced import Conv1d


class Generator(object):

    def __init__(self, args, embedding_layer):
        self.args = args
        self.embedding_layer = embedding_layer

    def ready(self):

        embedding_layer = self.embedding_layer

        args = self.args
        self.padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # inp_len x batch
        x = self.x = T.imatrix('x')
        fw_mask = self.fw_mask = T.imatrix('fw')
        chunk_sizes = self.chunk_sizes = T.imatrix('sizes')

        rv_mask = T.concatenate([T.ones((1, fw_mask.shape[1])), fw_mask[:-1]], axis=0)

        layers = self.layers = []

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        self.masks = masks_neq = T.cast(T.neq(chunk_sizes, 0), 'int32')
        masks_eq = T.cast(T.eq(chunk_sizes, 0), 'int32').dimshuffle((1, 0))

        embs = embedding_layer.forward(x.ravel())

        self.word_embs = embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        self.embs = apply_dropout(embs, dropout)

        if args.generator_encoding == 'cnn':
            h_final, size = self.cnn_encoding(chunk_sizes, rv_mask, n_e, n_d)
        else:
            h_final, size = self.lstm_encoding(fw_mask, rv_mask, n_e, n_d, activation)

        self.h_final = h_final = apply_dropout(h_final, dropout)

        output_layer = self.output_layer = ZLayer(
                n_in = size,
                n_hidden = args.hidden_dimension2,
                activation = activation,
                layer='rcnn',
            )

        z_pred, sample_updates = output_layer.sample_all(h_final)
        non_sampled_zpred, _ = output_layer.sample_all_pretrain(h_final)

        z_pred = theano.gradient.disconnected_grad(z_pred)

        z_pred_word_level, _ = theano.scan(fn=self.c_project,
                                           sequences=[z_pred.dimshuffle((1, 0)), chunk_sizes.dimshuffle((1, 0))]
                                           )
        non_sampled_zpred, _ = theano.scan(fn=self.c_project,
                                           sequences=[non_sampled_zpred.dimshuffle((1, 0)), chunk_sizes.dimshuffle((1, 0))]
                                           )

        self.sample_updates = sample_updates

        probs = output_layer.forward_all(h_final, z_pred)

        self.z_pred = z_pred_word_level = z_pred_word_level.dimshuffle((1, 0))
        self.non_sampled_zpred = non_sampled_zpred.dimshuffle((1, 0))

        logpz = - T.nnet.binary_crossentropy(probs, z_pred) * masks_neq
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape) * masks_neq

        # batch
        z = z_pred_word_level
        self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)

        params = self.params = [ ]
        for l in layers + [ output_layer ] + [embedding_layer]:
            for p in l.params:
                params.append(p)

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg
        self.l2_cost = l2_cost

    def pretrain(self):
        bm = self.bm = T.imatrix('bm')

        if self.args.bigram_m:
            padded = T.shape_padaxis(T.zeros_like(bm[0]), axis=1).dimshuffle((1, 0))
            bm_shift = T.concatenate([padded, bm[:-1]], axis=0)

            new_bm = T.cast(T.or_(bm, bm_shift), theano.config.floatX)
        else:
            new_bm = T.cast(bm, theano.config.floatX)

        new_probs = self.output_layer.forward_all(self.a_max_final, new_bm)

        cross_ent = T.nnet.binary_crossentropy(new_probs, new_bm) * self.masks
        self.obj = obj = T.mean(T.sum(cross_ent, axis=0))
        self.cost_g = obj * args.coeff_cost_scale + self.l2_cost

    def lstm_encoding(self, fw_mask, rv_mask, n_e, n_d, activation, layer_type='lstm'):
        layers = self.layers
        for i in xrange(2):
            l = LSTM(
                n_in=n_e,
                n_out=n_d,
                activation=activation,
                last_only=(i == 2)
            )
            layers.append(l)

        embs = self.embs

        flipped_embs = embs[::-1]

        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)[::-1]

        h1_red, _ = theano.scan(fn=self.c_reduce,
                                sequences=[h1.dimshuffle((1, 0, 2)), fw_mask.dimshuffle((1, 0))]
                                )
        h2_red, _ = theano.scan(fn=self.c_reduce,
                                sequences=[h2.dimshuffle((1, 0, 2)), rv_mask.dimshuffle((1, 0))]
                                )

        h1_red = h1_red.dimshuffle((1, 0, 2))
        h2_red = h2_red.dimshuffle((1, 0, 2))

        h_final = T.concatenate([h1_red, h2_red], axis=2)
        size = n_d * 2

        return h_final, size

    def cnn_encoding(self, chunk_sizes, rv_mask, n_e, n_d):
        window_sizes = [1, 3, 5, 7]
        pool_sizes = [2, 3, 4, 5]

        cnn_ls = []
        layers = self.layers

        embs = self.embs
        embs_for_c2d = embs.dimshuffle((1, 2, 0, 'x'))

        for c in xrange(len(window_sizes)):
            border = window_sizes[c] / 2

            conv_layer = Conv1d(n_in=n_e,
                                n_out=n_d,
                                window=window_sizes[c],
                                border_m=(border, 0))

            conv_out = conv_layer.forward(embs_for_c2d)
            conv_out_r = conv_out.reshape((conv_out.shape[0], conv_out.shape[1], conv_out.shape[2]))
            conv_out_r = conv_out_r.dimshuffle((0, 2, 1))

            cnn_ls.append(conv_out_r)
            layers.append(conv_layer)

        cnn_concat = T.concatenate(cnn_ls, axis=2)

        pool_out = []

        for p in xrange(len(pool_sizes)):
            pooled = pool_2d(cnn_concat, ws=(pool_sizes[p], 1), stride=(1, 1), ignore_border=True)

            z_shape = (pooled.shape[0], cnn_concat.shape[1] - pooled.shape[1], pooled.shape[2])
            zeros = T.zeros(shape=z_shape)

            padded_pooled = T.concatenate([pooled, zeros], axis=1)
            pool_out.append(padded_pooled)

        c_flat = chunk_sizes.dimshuffle((1, 0)).ravel()
        m_flat = rv_mask.dimshuffle((1, 0)).ravel()

        c_rep = T.repeat(c_flat, c_flat)
        c_rep = c_rep * m_flat

        all_chunks = [cnn_concat] + pool_out
        pooled_chunks = []
        size = n_d * len(window_sizes)

        for m in xrange(len(all_chunks)):
            c_mask = T.cast(T.eq(c_rep, m + 1), 'int32')
            c_mask_r = c_mask.reshape((1, c_mask.shape[0]))
            c_mask_tiled = T.tile(c_mask, (cnn_concat.shape[2], 1)).dimshuffle((1, 0))

            pooled_features = all_chunks[m].reshape(
                (all_chunks[m].shape[0] * all_chunks[m].shape[1], all_chunks[m].shape[2]))

            isolated_chunks = T.cast(pooled_features * c_mask_tiled, theano.config.floatX)
            pooled_chunks.append(isolated_chunks.reshape((embs.shape[1], embs.shape[0], size)))

        h = pooled_chunks[0] + pooled_chunks[1] + pooled_chunks[2]
        o1, _ = theano.scan(fn=self.c_reduce, sequences=[h, rv_mask.dimshuffle((1, 0))])

        h_final = o1.dimshuffle((1, 0, 2))

        return h_final, size

    def c_reduce(self, h, m):
        a = h[(m > 0).nonzero()]
        ze = T.zeros(shape=(h.shape[0] - a.shape[0], h.shape[1]))

        return T.concatenate([a, ze], axis=0)

    def c_project(self, h, m):
        return T.repeat(h, m)