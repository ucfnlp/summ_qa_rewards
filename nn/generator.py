import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

from nn.basic import apply_dropout
from nn.extended_layers import Sampler, LSTM
from nn.initialization import get_activation_by_name
from nn.advanced import Conv1d


class Generator(object):

    def __init__(self, args, embedding_layer, embedding_layer_posit):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embedding_layer_posit = embedding_layer_posit

    def ready(self):

        embedding_layer = self.embedding_layer

        args = self.args
        self.padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(np.float64(args.dropout).astype(theano.config.floatX))

        # inp_len x batch
        x = self.x = T.imatrix('x')
        fw_mask = self.fw_mask = T.imatrix('fw')
        chunk_sizes = self.chunk_sizes = T.imatrix('sizes')

        rv_mask = T.concatenate([T.ones((1, fw_mask.shape[1])), fw_mask[:-1]], axis=0)
        self.z_totals = T.sum(T.neq(self.x, self.padding_id), axis=0, dtype=theano.config.floatX)
        self.layers = []

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        self.pad_mask = T.cast(T.neq(x, self.padding_id), 'int32')
        self.chunk_mask = T.cast(T.neq(chunk_sizes, 0), 'int32')

        embs = embedding_layer.forward(x.ravel())

        self.word_embs = embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        self.embs = apply_dropout(embs, dropout)

        if args.generator_encoding == 'cnn':
            h_final, size = self.cnn_encoding(chunk_sizes, rv_mask, n_e, n_d/2)
        else:
            h_final, size = self.lstm_encoding(fw_mask, rv_mask, n_e, n_d, activation)

        self.size = size
        self.h_final = apply_dropout(h_final, dropout)

    def pretrain(self, inference):

        embedding_layer_posit = self.embedding_layer_posit

        bm = self.bm = T.imatrix('bm')
        posit_x = self.posit_x = T.imatrix('pos')

        embs_p = embedding_layer_posit.forward(posit_x.ravel())
        self.embs_p = embs_p = embs_p.reshape((bm.shape[0], bm.shape[1], embedding_layer_posit.n_d))

        reduced_p_embs, _ = theano.scan(fn=self.c_reduce,
                                        sequences=[embs_p.dimshuffle((1, 0, 2)), self.fw_mask.dimshuffle((1, 0))])
        reduced_p_embs = reduced_p_embs.dimshuffle((1, 0, 2))

        if not self.args.word_level_c:
            new_bm = T.cast(bm, theano.config.floatX)

            new_bm, _ = theano.scan(fn=self.bm_reduce,
                                    sequences=[new_bm.dimshuffle((1, 0)), self.fw_mask.dimshuffle((1, 0))])
            new_bm = new_bm.dimshuffle((1, 0))
        else:
            new_bm = T.cast(bm, theano.config.floatX)

        final_concat_d = self.size + embedding_layer_posit.n_d + 128 * 2
        output_rnn = Sampler(n_in=self.size,
                             n_out=128,
                             fc_in=final_concat_d,
                             fc_out=128,
                             sample=False)

        if inference:
            new_probs = output_rnn.s_forward_all(self.h_final, reduced_p_embs, inference)
        else:
            new_probs = output_rnn.pt_forward_all(self.h_final, reduced_p_embs, new_bm)

        new_probs = T.clip(new_probs, 1e-7, 1.0 - 1e-7)

        cross_ent = T.nnet.binary_crossentropy(new_probs, new_bm) * self.chunk_mask

        sz = T.cast(T.round(new_probs, mode='half_away_from_zero'), theano.config.floatX)

        z_pred_word_level, _ = theano.scan(fn=self.c_project,
                                           sequences=[sz.dimshuffle((1, 0)), self.chunk_sizes.dimshuffle((1, 0))]
                                           )
        self.non_sampled_zpred = z = z_pred_word_level.dimshuffle((1, 0))

        self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z[1:] - z[:-1]), axis=0, dtype=theano.config.floatX)

        self.layers.append(output_rnn)
        self.layers.append(output_rnn.fc_layer)
        self.layers.append(output_rnn.fc_layer_final)

        params = self.params = []
        for l in self.layers + [self.embedding_layer] + [self.embedding_layer_posit]:
            for p in l.params:
                params.append(p)

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p ** 2)
            else:
                l2_cost = l2_cost + T.sum(p ** 2)

        l2_cost = l2_cost * self.args.l2_reg

        self.l2_cost = l2_cost
        self.obj = obj = T.mean(T.sum(cross_ent, axis=0))
        self.cost_g = obj * self.args.coeff_cost_scale + self.l2_cost

    def sample(self, inference):
        embedding_layer_posit = self.embedding_layer_posit

        bm = self.bm = T.imatrix('bm')
        posit_x = self.posit_x = T.imatrix('pos')

        embs_p = embedding_layer_posit.forward(posit_x.ravel())
        self.embs_p = embs_p = embs_p.reshape((bm.shape[0], bm.shape[1], embedding_layer_posit.n_d))

        reduced_p_embs, _ = theano.scan(fn=self.c_reduce,
                                        sequences=[embs_p.dimshuffle((1, 0, 2)), self.fw_mask.dimshuffle((1, 0))])
        reduced_p_embs = reduced_p_embs.dimshuffle((1, 0, 2))

        final_concat_d = self.size + embedding_layer_posit.n_d + 128 * 2

        output_rnn = Sampler(n_in=self.size,
                             n_out=128,
                             fc_in=final_concat_d,
                             fc_out=128,
                             sample=True)
        if inference:
            probs = output_rnn.s_forward_all(self.h_final, reduced_p_embs, inference)
            samples = T.cast(T.round(probs, mode='half_away_from_zero'), theano.config.floatX)
        else:
            probs, updates, samples = output_rnn.s_forward_all(self.h_final, reduced_p_embs, inference)
            self.sample_updates = updates

        self.chunk_samples = samples

        z_pred_word_level, _ = theano.scan(fn=self.c_project,
                                           sequences=[samples.dimshuffle((1, 0)),
                                                      self.chunk_sizes.dimshuffle((1, 0))]
                                           )
        self.non_sampled_zpred = self.z_pred = z_pred_word_level = z_pred_word_level.dimshuffle((1, 0))
        self.logpz = - T.nnet.binary_crossentropy(probs, samples) * self.pad_mask
        self.probz = probs
        self.samps = samples

        self.zsum = T.sum(z_pred_word_level, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z_pred_word_level[1:] - z_pred_word_level[:-1]), axis=0, dtype=theano.config.floatX)

        self.layers.append(output_rnn)
        self.layers.append(output_rnn.fc_layer)
        self.layers.append(output_rnn.fc_layer_final)

        params = self.params = []
        for l in self.layers + [self.embedding_layer] + [self.embedding_layer_posit]:
            for p in l.params:
                params.append(p)

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p ** 2)
            else:
                l2_cost = l2_cost + T.sum(p ** 2)

        self.l2_cost = l2_cost * self.args.l2_reg

    def rl_out(self):
        z_pred_word_level = self.z_pred
        bm = self.bm

        bigram_ol = z_pred_word_level * bm

        total_z_bg_per_sample = T.sum(bigram_ol, axis=0)
        total_bg_per_sample = T.sum(bm, axis=0) + self.args.bigram_smoothing

        self.bigram_loss = bigram_loss = total_z_bg_per_sample / total_bg_per_sample
        self.zsum = T.abs_(self.zsum / self.z_totals - self.args.z_perc)

        self.zdiff = self.zdiff / self.z_totals
        self.cost_vec = cost_vec = self.args.coeff_adequacy * (1 - bigram_loss) + self.args.coeff_z * (
                2 * self.zsum + self.zdiff)

        self.cost_vec = cost_vec

        self.logpz = logpz = T.sum(self.logpz, axis=0)
        self.cost_logpz = cost_logpz = T.mean(cost_vec * logpz)

        self.obj = T.mean(T.sum(cost_vec, axis=0))
        self.cost_g = cost_logpz * self.args.coeff_cost_scale + self.l2_cost

    def lstm_encoding(self, fw_mask, rv_mask, n_e, n_d, activation):
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

        self.word_level_h = apply_dropout(T.concatenate([h1, h2], axis=2), self.dropout)

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

    def cnn_encoding(self, chunk_sizes, rv_mask, n_e, n_d=128):
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
        cnn_concat = T.nnet.relu(cnn_concat)

        pool_out = []

        for p in xrange(len(pool_sizes)):
            pooled = pool_2d(cnn_concat, ws=(pool_sizes[p], 1), stride=(1, 1), ignore_border=True)

            z_shape = (pooled.shape[0], cnn_concat.shape[1] - pooled.shape[1], pooled.shape[2])
            zeros = T.zeros(shape=z_shape)

            padded_pooled = T.concatenate([pooled, zeros], axis=1)
            pool_out.append(padded_pooled)

        c_flat = chunk_sizes.dimshuffle((1, 0))
        m_flat = rv_mask.dimshuffle((1, 0)).ravel()

        c_rep, _ = theano.scan(fn=self.c_project,
                               sequences=[c_flat,
                                          c_flat])

        c_rep = c_rep.ravel() * m_flat

        all_chunks = [cnn_concat] + pool_out
        pooled_chunks = []
        size = n_d * len(window_sizes)

        for m in xrange(len(all_chunks)):
            c_mask = T.cast(T.eq(c_rep, m + 1), 'int32')
            c_mask_tiled = T.tile(c_mask, (cnn_concat.shape[2], 1)).dimshuffle((1, 0))

            pooled_features = all_chunks[m].reshape(
                (all_chunks[m].shape[0] * all_chunks[m].shape[1], all_chunks[m].shape[2]))

            isolated_chunks = T.cast(pooled_features * c_mask_tiled, theano.config.floatX)
            pooled_chunks.append(isolated_chunks.reshape((embs.shape[1], embs.shape[0], size)))

        h = pooled_chunks[0] + pooled_chunks[1] + pooled_chunks[2] + pooled_chunks[3] + pooled_chunks[4]
        o1, _ = theano.scan(fn=self.c_reduce, sequences=[h, rv_mask.dimshuffle((1, 0))])

        h_final = o1.dimshuffle((1, 0, 2))

        return h_final, size

    def c_reduce(self, h, m):
        a = h[(m > 0).nonzero()]
        ze = T.zeros(shape=(h.shape[0] - a.shape[0], h.shape[1]))

        return T.concatenate([a, ze], axis=0)

    def bm_reduce(self, h, m):
        a = h[(m > 0).nonzero()]
        ze = T.zeros(shape=(h.shape[0] - a.shape[0],))

        return T.concatenate([a, ze], axis=0)

    def c_project(self, h, m):
        valid_projection_truncated = T.repeat(h, m)
        zs = T.zeros(shape=(h.shape[0] - valid_projection_truncated.shape[0],))

        return T.concatenate([valid_projection_truncated, zs], axis=0)