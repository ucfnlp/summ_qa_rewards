import theano
import theano.tensor as T

from nn.basic import Layer, apply_dropout
from nn.extended_layers import HLLSTM
from nn.initialization import softmax, get_activation_by_name
from nn.advanced import Bilinear


class Encoder(object):

    def __init__(self, args, nclasses, generator):
        self.args = args
        self.embedding_layer = generator.embedding_layer
        self.nclasses = nclasses
        self.generator = generator

    def ready(self):
        generator = self.generator
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        layers = []
        params = self.params = []

        # hl_inp_len x (batch * n)
        y = self.y = T.imatrix('y')
        # (batch * n) x n_classes
        gold_standard_entities = self.gold_standard_entities = T.imatrix('gs')
        # inp_len x batch
        bm = self.bm = generator.bm

        loss_mask = self.loss_mask = T.imatrix('loss_mask')

        # inp_len x batch
        x = generator.x
        z = generator.z_pred

        mask_y = T.cast(T.neq(y, padding_id), theano.config.floatX).dimshuffle((0, 1, 'x'))

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d

        rnn_fw = HLLSTM(
            n_in=n_e,
            n_out=n_d
        )

        rnn_rv = HLLSTM(
            n_in=n_e,
            n_out=n_d
        )

        embs_y = embedding_layer.forward(y.ravel())
        embs_y = embs_y.reshape((y.shape[0], y.shape[1], n_e))

        embs_x = apply_dropout(generator.word_embs, generator.dropout)

        flipped_embs_y = embs_y[::-1]
        flipped_mask_y = mask_y[::-1]

        h_f_y = rnn_fw.forward_all(embs_y, mask_y)
        h_r_y = rnn_rv.forward_all(flipped_embs_y, flipped_mask_y)

        if args.use_generator_h:
            mask_x = self.generator.chunk_samples * self.generator.chunk_mask
            mask_x = mask_x.dimshuffle((0, 1, 'x'))
            tiled_x_mask = T.tile(mask_x, (args.n, 1)).dimshuffle((1, 0, 2))

            h_concat_x = self.generator.h_final

            if args.generator_encoding == 'cnn':
                layers.extend(self.generator.layers[:4])
            else:
                layers.extend(self.generator.layers[:2])
        else:
            mask_x = T.cast(T.neq(x, padding_id) * z, theano.config.floatX).dimshuffle((0, 1, 'x'))
            tiled_x_mask = T.tile(mask_x, (args.n, 1)).dimshuffle((1, 0, 2))

            flipped_embs_x = embs_x[::-1]
            flipped_mask_x = mask_x[::-1]

            h_f_x = rnn_fw.forward_all_x(embs_x, mask_x)
            h_r_x = rnn_rv.forward_all_x(flipped_embs_x, flipped_mask_x)

            h_concat_x = T.concatenate([h_f_x, h_r_x[::-1]], axis=2)

        softmax_mask = T.zeros_like(tiled_x_mask) - 1e8
        softmax_mask = softmax_mask * (tiled_x_mask - 1)

        # 1 x (batch * n) x n_d -> (batch * n) x (2 * n_d) x 1
        h_concat_y = T.concatenate([h_f_y, h_r_y], axis=2).dimshuffle((1, 2, 0))
        h_concat_y = apply_dropout(h_concat_y, generator.dropout)

        # inp_len x batch x n_d -> inp_len x batch x (2 * n_d)
        # (batch * n) x inp_len x (2 * n_d)
        gen_h_final = T.tile(h_concat_x, (args.n, 1)).dimshuffle((1, 0, 2))

        if args.bilinear:
            bilinear_l = Bilinear(n_d, x.shape[1], args.n)
            inp_dot_hl = bilinear_l.forward(gen_h_final, h_concat_y)

            layers.append(bilinear_l)
        else:
            # (batch * n) x inp_len x 1
            inp_dot_hl = T.batched_dot(gen_h_final, h_concat_y)

        size = n_d * 2

        inp_dot_hl = inp_dot_hl - softmax_mask
        inp_dot_hl = inp_dot_hl.ravel()

        # (batch * n) x inp_len
        self.alpha = alpha = T.nnet.softmax(inp_dot_hl.reshape((args.n * x.shape[1], x.shape[0])))

        # (batch * n) x n_d * 2
        o = T.batched_dot(alpha, gen_h_final)

        size *= 4
        h_concat_y = h_concat_y.reshape((o.shape[0], o.shape[1]))
        self.o = o = T.concatenate([o, h_concat_y, T.abs_(o - h_concat_y), o * h_concat_y], axis=1)

        fc7 = Layer(
            n_in=size,
            n_out=size,
            activation=get_activation_by_name('relu'),
            has_bias=True
        )
        fc7_out = fc7.forward(o)

        output_layer = Layer(
            n_in=size,
            n_out=self.nclasses,
            activation=softmax,
            has_bias=True
        )

        layers.append(rnn_fw)
        layers.append(rnn_rv)
        layers.append(fc7)
        layers.append(output_layer)

        preds = output_layer.forward(fc7_out + o)
        self.preds_clipped = preds_clipped = T.clip(preds, 1e-7, 1.0 - 1e-7)

        cross_entropy = T.nnet.categorical_crossentropy(preds_clipped, gold_standard_entities)
        loss_mat = cross_entropy.reshape((x.shape[1], args.n))

        loss_mat = loss_mat * loss_mask

        word_ol = z * bm

        total_z_word_overlap_per_sample = T.sum(word_ol, axis=0)
        total_overlap_per_sample = T.sum(bm, axis=0) + args.bigram_smoothing

        self.word_overlap_loss = word_overlap_loss = total_z_word_overlap_per_sample / total_overlap_per_sample

        self.loss_vec = loss_vec = T.mean(loss_mat, axis=1)

        logpz = generator.logpz

        loss = self.loss = T.mean(loss_vec)

        z_totals = T.sum(T.neq(x, padding_id), axis=0, dtype=theano.config.floatX)
        self.zsum = zsum = T.abs_(generator.zsum / z_totals - args.z_perc)
        self.zdiff = zdiff = generator.zdiff / z_totals

        self.cost_vec = cost_vec = loss_vec + args.coeff_adequacy * (1 - word_overlap_loss) + args.coeff_z * (
                    2 * zsum + zdiff)

        self.logpz = logpz = T.sum(logpz, axis=0)
        self.cost_logpz = cost_logpz = T.mean(cost_vec * logpz)
        self.obj = T.mean(cost_vec)

        for l in layers + [embedding_layer]:
            for p in l.params:
                params.append(p)

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p ** 2)
            else:
                l2_cost = l2_cost + T.sum(p ** 2)

        l2_cost = l2_cost * args.l2_reg
        self.l2_cost = l2_cost

        self.cost_g = cost_logpz * args.coeff_cost_scale + generator.l2_cost
        self.cost_e = loss + l2_cost
