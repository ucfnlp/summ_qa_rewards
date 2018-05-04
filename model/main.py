import cPickle as pickle
import gzip
import json
import os
import time
import random
from datetime import datetime

import numpy as np
import sklearn.metrics as sk
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

import myio
import summarization_args

from nn.basic import LSTM, apply_dropout, Layer
from nn.extended_layers import ZLayer, HLLSTM, RCNN
from nn.initialization import get_activation_by_name, softmax, create_shared, random_init
from nn.optimization import create_optimization_updates
from nn.advanced import Bilinear, Conv1d
from util import say


class Generator(object):

    def __init__(self, args, embedding_layer):
        self.args = args
        self.embedding_layer = embedding_layer

    def ready(self):

        def c_reduce(h, m):
            a = h[(m > 0).nonzero()]
            ze = T.zeros(shape=(h.shape[0] - a.shape[0], h.shape[1]))

            return T.concatenate([a, ze], axis=0)

        def c_project(h, m):
            return T.repeat(h, m)

        def repeat_2d(a_max, tile_amt):
            return T.tile(a_max, tile_amt)

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

        window_sizes = [1, 3, 5, 7]
        pool_sizes = [2, 3, 4, 5]

        cnn_ls = []
        layers = self.layers = []

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)
        size = n_d * len(window_sizes)

        self.masks = masks_neq = T.cast(T.neq(chunk_sizes, 0), 'int32')
        masks_eq = T.cast(T.eq(chunk_sizes, 0), 'int32').dimshuffle((1, 0))

        embs = embedding_layer.forward(x.ravel())

        self.word_embs = embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)
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
            pooled = pool_2d(cnn_concat, ws=(pool_sizes[p], 1), stride=(1, 1))

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

        for m in xrange(len(all_chunks)):
            c_mask = T.cast(T.eq(c_rep, m + 1), 'int32')
            c_mask_r = c_mask.reshape((1, c_mask.shape[0]))
            c_mask_tiled = T.tile(c_mask, (cnn_concat.shape[2], 1)).dimshuffle((1, 0))

            pooled_features = all_chunks[m].reshape(
                (all_chunks[m].shape[0] * all_chunks[m].shape[1], all_chunks[m].shape[2]))

            isolated_chunks = T.cast(pooled_features * c_mask_tiled, theano.config.floatX)
            pooled_chunks.append(isolated_chunks.reshape((embs.shape[1], embs.shape[0], size)))

        h = pooled_chunks[0] + pooled_chunks[1] + pooled_chunks[2]
        o1, _ = theano.scan(fn=c_reduce, sequences=[h, rv_mask.dimshuffle((1, 0))])
        h_final = o1.dimshuffle((1, 0, 2))

        output_layer = self.output_layer = ZLayer(
                n_in = size,
                n_hidden = args.hidden_dimension2,
                activation = activation,
                layer='rcnn',
            )

        z_pred, sample_updates = output_layer.sample_all(h_final)
        non_sampled_zpred, _ = output_layer.sample_all_pretrain(h_final)

        z_pred = theano.gradient.disconnected_grad(z_pred)

        z_pred_word_level, _ = theano.scan(fn=c_project,
                                           sequences=[z_pred.dimshuffle((1, 0)), chunk_sizes.dimshuffle((1, 0))]
                                           )
        non_sampled_zpred, _ = theano.scan(fn=c_project,
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
        bm = self.bm = T.imatrix('bm')

        # mask for loss
        if not args.pad_repeat:
            loss_mask = self.loss_mask = T.imatrix('loss_mask')

        # inp_len x batch
        x = generator.x
        z = generator.z_pred

        mask_x = T.cast(T.neq(x, padding_id) * z, theano.config.floatX).dimshuffle((0,1,'x'))
        tiled_x_mask = T.tile(mask_x, (args.n, 1)).dimshuffle((1, 0, 2))
        mask_y = T.cast(T.neq(y, padding_id), theano.config.floatX).dimshuffle((0, 1, 'x'))

        softmax_mask = T.zeros_like(tiled_x_mask) - 1e8
        softmax_mask = softmax_mask * (tiled_x_mask - 1)

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

        embs_x = generator.word_embs

        flipped_embs_x = embs_x[::-1]
        flipped_mask_x = mask_x[::-1]

        flipped_embs_y = embs_y[::-1]
        flipped_mask_y = mask_y[::-1]

        h_f_y = rnn_fw.forward_all(embs_y, mask_y)
        h_r_y = rnn_rv.forward_all(flipped_embs_y, flipped_mask_y)

        h_f_x = rnn_fw.forward_all_x(embs_x, mask_x)
        h_r_x = rnn_rv.forward_all_x(flipped_embs_x, flipped_mask_x)

        # 1 x (batch * n) x n_d -> (batch * n) x (2 * n_d) x 1
        h_concat_y = T.concatenate([h_f_y, h_r_y], axis=2).dimshuffle((1, 2, 0))

        # inp_len x batch x n_d -> inp_len x batch x (2 * n_d)
        h_concat_x = T.concatenate([h_f_x, h_r_x[::-1]], axis=2)

        # (batch * n) x inp_len x (2 * n_d)
        gen_h_final = T.tile(h_concat_x, (args.n, 1)).dimshuffle((1, 0, 2))

        if args.bilinear:
            bilinear_l = Bilinear(n_d, x.shape[1], args.n)
            inp_dot_hl = bilinear_l.forward(gen_h_final, h_concat_y)

            params.append(bilinear_l.params[0])
        else:
            # (batch * n) x inp_len x 1
            inp_dot_hl = T.batched_dot(gen_h_final, h_concat_y)

        size = n_d * 2

        inp_dot_hl = inp_dot_hl - softmax_mask
        inp_dot_hl = inp_dot_hl.ravel()

        # (batch * n) x inp_len
        self.alpha = alpha = T.nnet.softmax(inp_dot_hl.reshape((args.n * x.shape[1], x.shape[0])))

        # (batch * n) * n_d * 2
        o = T.batched_dot(alpha, gen_h_final)

        if self.args.extended_c_k:
            size *= 4
            h_concat_y = h_concat_y.reshape((o.shape[0], o.shape[1]))
            self.o = o = T.concatenate([o, h_concat_y, T.abs_(o - h_concat_y), o * h_concat_y], axis=1)

        output_layer = Layer(
            n_in=size,
            n_out=self.nclasses,
            activation=softmax
        )

        layers.append(rnn_fw)
        layers.append(rnn_rv)
        layers.append(output_layer)

        preds = output_layer.forward(o)
        self.preds_clipped = preds_clipped = T.clip(preds, 1e-7, 1.0 - 1e-7)

        cross_entropy = T.nnet.categorical_crossentropy(preds_clipped, gold_standard_entities)
        loss_mat = cross_entropy.reshape((x.shape[1], args.n))

        if not args.pad_repeat:
            loss_mat = loss_mat * loss_mask

        z_shift = z[1:]
        z_new = z[:-1]

        if self.args.bigram_m:
            valid_bg = z_new * z_shift
            bigram_ol = valid_bg * bm[:-1]
        else:
            bigram_ol = z * bm

        total_z_bg_per_sample = T.sum(bigram_ol, axis=0)
        total_bg_per_sample = T.sum(bm, axis=0) + args.bigram_smoothing

        self.bigram_loss = bigram_loss = total_z_bg_per_sample / total_bg_per_sample

        self.loss_vec = loss_vec = T.mean(loss_mat, axis=1)

        self.zsum = zsum = generator.zsum
        self.zdiff = zdiff = generator.zdiff
        logpz = generator.logpz

        loss = self.loss = T.mean(loss_vec)

        z_totals = T.sum(T.neq(x, padding_id), axis=0, dtype=theano.config.floatX)
        self.zsum = zsum = T.abs_(zsum / z_totals - args.z_perc)
        self.zdiff = zdiff = zdiff / z_totals

        self.cost_vec = cost_vec = loss_vec + args.coeff_adequacy * (1 - bigram_loss) + args.coeff_z * (
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
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg
        self.l2_cost = l2_cost

        self.cost_g = cost_logpz * args.coeff_cost_scale + generator.l2_cost
        self.cost_e = loss  + l2_cost


class Model(object):
    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses

        self.generator = Generator(args, embedding_layer)
        self.encoder = Encoder(args, nclasses, self.generator)

        self.generator.ready()
        self.encoder.ready()

        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.fw_mask = self.generator.fw_mask
        self.chunk_sizes = self.generator.chunk_sizes

        self.y = self.encoder.y
        self.bm = self.encoder.bm
        self.gold_standard_entities = self.encoder.gold_standard_entities

        self.z = self.generator.z_pred

        self.params = self.encoder.params + self.generator.params

    def ready_test(self):
        args, embedding_layer = self.args, self.embedding_layer
        self.generator = Generator(args, embedding_layer)

        self.generator.ready()
        self.dropout = self.generator.dropout

        self.x = self.generator.x
        self.fw_mask = self.generator.fw_mask
        self.chunk_sizes = self.generator.chunk_sizes
        self.z = self.generator.non_sampled_zpred

    def ready_pretrain(self):
        args, embedding_layer = self.args, self.embedding_layer
        self.generator = Generator(args, embedding_layer)

        self.generator.ready()
        self.generator.pretrain()

        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.bm = self.generator.bm
        self.fw_mask = self.generator.fw_mask
        self.chunk_sizes = self.generator.chunk_sizes

        self.z = self.generator.non_sampled_zpred
        self.params = self.generator.params

    def save_model(self, path, args, pretrain=False):
        # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        # output to path
        with gzip.open(path, "wb") as fout:

            if pretrain:
                pickle.dump(
                    ([x.get_value() for x in self.generator.params],  # generator
                     self.nclasses,
                     args  # training configuration
                     ),
                    fout,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            else:
                pickle.dump(
                    ([x.get_value() for x in self.encoder.params],  # encoder
                     [x.get_value() for x in self.generator.params],  # generator
                     self.nclasses,
                     args  # training configuration
                     ),
                    fout,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        if args.trained_emb:
            fname = args.trained_emb + ('pretrain/' if pretrain else '') + myio.create_fname_identifier(args) + '.txt'
            ofp = open(fname, 'w+')
            vectors = self.embedding_layer.params[0].get_value()
            emb_len = args.embedding_dim

            for i in xrange(len(self.embedding_layer.lst_words)):
                word = self.embedding_layer.lst_words[i]
                emb = vectors[i]

                ofp.write(word + ' ')

                for v in xrange(emb_len):
                    ofp.write(str(emb[v]))

                    if v == emb_len - 1:
                        ofp.write('\n')
                    else:
                        ofp.write(' ')

            ofp.close()

    def load_model(self, path, test=False):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            eparams, gparams, nclasses, args = pickle.load(fin)

        # construct model/network using saved configuration
        self.args = args
        self.nclasses = nclasses

        if test:
            self.ready_test()
        else:
            self.ready()
            for x, v in zip(self.encoder.params, eparams):
                x.set_value(v)

        for x, v in zip(self.generator.params, gparams):
            x.set_value(v)

    def load_model_pretrain(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            gparams, nclasses, args = pickle.load(fin)

        if self.args.pretrain:
            self.args = args
            self.nclasses = nclasses
            self.ready_pretrain()
        else:
            self.ready()

        for x, v in zip(self.generator.params, gparams):
            x.set_value(v)

    def test(self):
        args = self.args

        test_generator = theano.function(
            inputs=[self.x, self.fw_mask, self.chunk_sizes],
            outputs=self.z,
            updates=self.generator.sample_updates
        )

        self.dropout.set_value(0.0)
        z, x, y, e, sha = self.evaluate_test_data(test_generator)

        myio.save_test_results_rouge(args, z, x, y, e, sha, self.embedding_layer)

    def dev(self):

        eval_generator = theano.function(
            inputs=[self.x, self.bm, self.fw_mask, self.chunk_sizes],
            outputs=[self.z, self.generator.cost_g, self.generator.obj],
            updates=self.generator.sample_updates,
            on_unused_input='ignore'
        )

        self.dropout.set_value(0.0)

        self.evaluate_pretrain_data_rouge(eval_generator)
        myio.get_rouge(self.args)

    def dev_full(self):

        inputs_d = [self.x, self.y, self.bm, self.gold_standard_entities, self.fw_mask, self.chunk_sizes]
        if not args.pad_repeat:
            inputs_d.append(self.encoder.loss_mask)

        eval_generator = theano.function(
            inputs=inputs_d,
            outputs=[self.generator.non_sampled_zpred, self.encoder.obj, self.encoder.loss, self.encoder.preds_clipped],
            updates=self.generator.sample_updates,
            on_unused_input='ignore'
        )

        self.dropout.set_value(0.0)

        dev_obj, dev_z, dev_x, dev_sha, dev_acc = self.evaluate_data(eval_generator)
        myio.save_dev_results(self.args, None, dev_z, dev_x, dev_sha)
        myio.get_rouge(self.args)

    def train(self):
        args = self.args
        dropout = self.dropout
        padding_id = self.embedding_layer.vocab_map["<padding>"]

        updates_e, lr_e, gnorm_e = create_optimization_updates(
            cost=self.encoder.cost_e,
            params=self.encoder.params,
            method=args.learning,
            beta1=args.beta1,
            beta2=args.beta2,
            lr=args.learning_rate
        )[:3]

        updates_g, lr_g, gnorm_g = create_optimization_updates(
            cost=self.encoder.cost_g,
            params=self.generator.params,
            method=args.learning,
            beta1=args.beta1,
            beta2=args.beta2,
            lr=args.learning_rate
        )[:3]

        outputs_d = [self.generator.non_sampled_zpred, self.encoder.obj, self.encoder.loss, self.encoder.preds_clipped]
        outputs_t = [self.encoder.obj, self.encoder.loss, self.z, self.encoder.zsum, self.encoder.zdiff,
                     self.encoder.bigram_loss, self.encoder.loss_vec, self.encoder.cost_logpz, self.encoder.logpz,
                     self.encoder.cost_vec, self.generator.masks, self.encoder.bigram_loss, self.encoder.preds_clipped,
                     self.encoder.alpha, self.encoder.o]

        inputs_d = [self.x, self.y, self.bm, self.gold_standard_entities, self.fw_mask, self.chunk_sizes]
        inputs_t = [self.x, self.y, self.bm, self.gold_standard_entities, self.fw_mask, self.chunk_sizes]

        if not args.pad_repeat:
            inputs_d.append(self.encoder.loss_mask)
            inputs_t.append(self.encoder.loss_mask)

        eval_generator = theano.function(
            inputs=inputs_d,
            outputs=outputs_d,
            updates=self.generator.sample_updates,
            on_unused_input='ignore'
        )

        train_generator = theano.function(
            inputs=inputs_t,
            outputs=outputs_t,
            updates=updates_e.items() + updates_g.items() + self.generator.sample_updates,
            on_unused_input='ignore'
        )

        unchanged = 0
        best_dev = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.10 + 1e-3
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        filename = myio.create_json_filename(args)
        ofp_train = open(filename, 'w+')
        json_train = dict()

        for epoch in xrange(args.max_epochs):
            unchanged += 1
            more_count = 0

            say("Unchanged : {}\n".format(unchanged))

            if unchanged > 25:
                break

            more = True
            if args.decay_lr:
                param_bak = [p.get_value(borrow=False) for p in self.params]

            while more:
                train_cost = 0.0
                train_loss = 0.0
                p1 = 0.0
                more_count += 1

                if more_count > 5:
                    break
                start_time = time.time()

                loss_all = []
                obj_all = []
                zsum_all = []
                bigram_loss_all = []
                loss_vec_all = []
                z_diff_all = []
                cost_logpz_all = []
                logpz_all = []
                z_pred_all = []
                cost_vec_all = []
                train_acc = []

                num_files = args.num_files_train
                N = args.online_batch_size * num_files

                for i in xrange(num_files):
                    if args.pad_repeat:
                        train_batches_x, train_batches_y, train_batches_e, train_batches_bm, _, train_batches_fw, train_batches_csz = myio.load_batches(
                            args.batch_dir + args.source + 'train', i)
                    else:
                        train_batches_x, train_batches_y, train_batches_e, train_batches_bm, train_batches_blm, _, train_batches_fw, train_batches_csz = myio.load_batches(
                            args.batch_dir + args.source + 'train', i)

                    cur_len = len(train_batches_x)

                    random.seed(datetime.now())
                    perm2 = range(cur_len)
                    random.shuffle(perm2)

                    train_batches_x = [train_batches_x[k] for k in perm2]
                    train_batches_y = [train_batches_y[k] for k in perm2]
                    train_batches_e = [train_batches_e[k] for k in perm2]
                    train_batches_bm = [train_batches_bm[k] for k in perm2]
                    train_batches_fw = [train_batches_fw[k] for k in perm2]
                    train_batches_csz = [train_batches_csz[k] for k in perm2]

                    if not args.pad_repeat:
                        train_batches_blm = [train_batches_blm[k] for k in perm2]

                    for j in xrange(cur_len):
                        if args.full_test:
                            if (i* args.online_batch_size + j + 1) % 10 == 0:
                                say("\r{}/{} {:.2f}       ".format(i* args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))
                        elif (i* args.online_batch_size + j + 1) % 10 == 0:
                                say("\r{}/{} {:.2f}       ".format(i* args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))

                        if args.pad_repeat:
                            bx, by, be, bm, bfw, bcsz = train_batches_x[j], train_batches_y[j], train_batches_e[j], \
                                                  train_batches_bm[j], train_batches_fw[j], train_batches_csz[j]
                            cost, loss, z, zsum, zdiff, bigram_loss, loss_vec, cost_logpz, logpz, cost_vec, masks, bigram_loss, preds_tr = train_generator(
                                bx, by, bm, be, bfw, bcsz)
                        else:
                            bx, by, be, bm, blm, bfw, bcsz = train_batches_x[j], train_batches_y[j], train_batches_e[j], \
                                                  train_batches_bm[j], train_batches_blm[j], train_batches_fw[j], train_batches_csz[j]

                            cost, loss, z, zsum, zdiff, bigram_loss, loss_vec, cost_logpz, logpz, cost_vec, masks, bigram_loss, preds_tr, alpha, o = train_generator(
                                bx, by, bm, be, bfw, bcsz, blm)

                        mask = bx != padding_id

                        train_acc.append(self.eval_acc(be, preds_tr))
                        obj_all.append(cost)
                        loss_all.append(loss)
                        zsum_all.append(np.mean(zsum))
                        loss_vec_all.append(np.mean(loss_vec))
                        z_diff_all.append(np.mean(zdiff))
                        cost_logpz_all.append(np.mean(cost_logpz))
                        logpz_all.append(np.mean(logpz))
                        z_pred_all.append(np.mean(np.sum(z, axis=0)))
                        cost_vec_all.append(np.mean(cost_vec))
                        bigram_loss_all.append(np.mean(bigram_loss))

                        train_cost += cost
                        train_loss += loss

                        p1 += np.sum(z * mask) / (np.sum(mask) + 1e-8)

                cur_train_avg_cost = train_cost / N

                if args.dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_z, dev_x, dev_sha, dev_acc = self.evaluate_data(eval_generator)
                    self.dropout.set_value(dropout_prob)
                    cur_dev_avg_cost = dev_obj

                more = False

                if args.decay_lr and last_train_avg_cost is not None:
                    if cur_train_avg_cost > last_train_avg_cost * (1 + tolerance):
                        more = True
                        say("\nTrain cost {} --> {}\n".format(
                            last_train_avg_cost, cur_train_avg_cost
                        ))
                    if args.dev and cur_dev_avg_cost > last_dev_avg_cost * (1 + tolerance):
                        more = True
                        say("\nDev cost {} --> {}\n".format(
                            last_dev_avg_cost, cur_dev_avg_cost
                        ))

                if more:
                    lr_val = lr_g.get_value() * 0.5
                    lr_val = np.float64(lr_val).astype(theano.config.floatX)
                    lr_g.set_value(lr_val)
                    lr_e.set_value(lr_val)
                    say("Decrease learning rate to {}\n".format(float(lr_val)))
                    for p, v in zip(self.params, param_bak):
                        p.set_value(v)
                    continue

                myio.record_observations_verbose(json_train, epoch + 1, loss_all, obj_all, zsum_all, loss_vec_all,
                                             z_diff_all, cost_logpz_all, logpz_all, z_pred_all, cost_vec_all, bigram_loss_all, dev_acc, np.mean(train_acc))

                last_train_avg_cost = cur_train_avg_cost

                say("\n")
                say(("Generator Epoch {:.2f}  costg={:.4f}  lossg={:.4f}  " +
                     "\t[{:.2f}m / {:.2f}m]\n").format(
                    epoch,
                    train_cost / N,
                    train_loss / N,
                    (time.time() - start_time) / 60.0,
                    (time.time() - start_time) / 60.0 / (i * args.online_batch_size + j + 1) * N
                ))

                if args.dev:
                    last_dev_avg_cost = cur_dev_avg_cost
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.save_model:
                            filename = args.save_model + myio.create_fname_identifier(args)
                            self.save_model(filename, args)
                            json_train['BEST_DEV_EPOCH'] = epoch

                            myio.save_dev_results(self.args, None, dev_z, dev_x, dev_sha)

            if more_count > 5:
                json_train['ERROR'] = 'Stuck reducing error rate, at epoch ' + str(epoch + 1) + '. LR = ' + str(lr_val)
                json.dump(json_train, ofp_train)
                ofp_train.close()
                return

        if unchanged > 20:
            json_train['UNCHANGED'] = unchanged

        json.dump(json_train, ofp_train)
        ofp_train.close()

    def pretrain(self):
        args = self.args
        padding_id = self.embedding_layer.vocab_map["<padding>"]

        updates_g, lr_g, gnorm_g = create_optimization_updates(
            cost=self.generator.cost_g,
            params=self.generator.params,
            method=args.learning,
            beta1=args.beta1,
            beta2=args.beta2,
            lr=args.learning_rate
        )[:3]

        eval_generator = theano.function(
            inputs=[self.x,  self.bm, self.fw_mask, self.chunk_sizes],
            outputs=[self.z, self.generator.cost_g, self.generator.obj],
            updates=self.generator.sample_updates
        )

        train_generator = theano.function(
            inputs=[self.x, self.bm, self.fw_mask, self.chunk_sizes],
            outputs=[self.generator.obj, self.z, self.generator.zsum, self.generator.zdiff,  self.generator.cost_g],
            updates=updates_g.items() + self.generator.sample_updates
        )

        unchanged = 0
        best_dev = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.10 + 1e-3
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        filename = myio.create_json_filename(args)
        ofp_train = open(filename, 'w+')
        json_train = dict()

        for epoch in xrange(args.max_epochs):
            unchanged += 1
            more_count = 0

            say("Unchanged : {}\n".format(unchanged))

            if unchanged > 20:
                break

            more = True
            if args.decay_lr:
                param_bak = [p.get_value(borrow=False) for p in self.params]

            while more:
                train_cost = 0.0
                train_loss = 0.0
                p1 = 0.0
                more_count += 1

                if more_count > 5:
                    break
                start_time = time.time()

                obj_all = []
                zsum_all = []
                z_diff_all = []
                z_pred_all = []

                num_files = args.num_files_train
                N = args.online_batch_size * num_files

                for i in xrange(num_files):
                    if args.pad_repeat:
                        train_batches_x, _, _, train_batches_bm,  _, train_batches_fw, train_batches_csz = myio.load_batches(
                            args.batch_dir + args.source + 'train', i)
                    else:
                        train_batches_x, _, _, train_batches_bm, _, _, train_batches_fw, train_batches_csz = myio.load_batches(
                            args.batch_dir + args.source + 'train', i)

                    random.seed(datetime.now())
                    perm2 = range(len(train_batches_x))
                    random.shuffle(perm2)

                    train_batches_x = [train_batches_x[k] for k in perm2]
                    train_batches_bm = [train_batches_bm[k] for k in perm2]
                    train_batches_fw = [train_batches_fw[k] for k in perm2]
                    train_batches_csz = [train_batches_csz[k] for k in perm2]

                    cur_len = len(train_batches_x)

                    for j in xrange(cur_len):
                        if args.full_test:
                            if (i * args.online_batch_size + j + 1) % 10 == 0:
                                say("\r{}/{} {:.2f}       ".format(i * args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))
                        elif (i * args.online_batch_size + j + 1) % 10 == 0:
                            say("\r{}/{} {:.2f}       ".format(i * args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))

                        bx, bm, bfw, bcz = train_batches_x[j], train_batches_bm[j], train_batches_fw[j], train_batches_csz[j]
                        # print bx.shape, bm.shape
                        mask = bx != padding_id

                        obj, z, zsum, zdiff,cost_g = train_generator(bx, bm, bfw, bcz)
                        zsum_all.append(np.mean(zsum))
                        z_diff_all.append(np.mean(zdiff))
                        z_pred_all.append(np.mean(np.sum(z, axis=0)))
                        obj_all.append(np.mean(obj))

                        train_cost += obj

                        p1 += np.sum(z * mask) / (np.sum(mask) + 1e-8)

                cur_train_avg_cost = train_cost / N

                if args.dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_z, x, sha_ls = self.evaluate_pretrain_data(eval_generator)
                    self.dropout.set_value(dropout_prob)
                    cur_dev_avg_cost = dev_obj

                more = False

                if args.decay_lr and last_train_avg_cost is not None:
                    if cur_train_avg_cost > last_train_avg_cost * (1 + tolerance):
                        more = True
                        say("\nTrain cost {} --> {}\n".format(
                            last_train_avg_cost, cur_train_avg_cost
                        ))
                    if args.dev and cur_dev_avg_cost > last_dev_avg_cost * (1 + tolerance):
                        more = True
                        say("\nDev cost {} --> {}\n".format(
                            last_dev_avg_cost, cur_dev_avg_cost
                        ))

                if more:
                    lr_val = lr_g.get_value() * 0.5
                    lr_val = np.float64(lr_val).astype(theano.config.floatX)
                    lr_g.set_value(lr_val)
                    say("Decrease learning rate to {}\n".format(float(lr_val)))
                    for p, v in zip(self.params, param_bak):
                        p.set_value(v)
                    continue

                myio.record_observations_pretrain(json_train, epoch + 1, obj_all, zsum_all, z_diff_all, z_pred_all)

                last_train_avg_cost = cur_train_avg_cost

                say("\n")
                say(("Generator Epoch {:.2f}  costg={:.4f}  lossg={:.4f}  " +
                     "\t[{:.2f}m / {:.2f}m]\n").format(
                    epoch,
                    train_cost / N,
                    train_loss / N,
                    (time.time() - start_time) / 60.0,
                    (time.time() - start_time) / 60.0 / (i * args.online_batch_size + j + 1) * N
                ))

                if args.dev:
                    last_dev_avg_cost = cur_dev_avg_cost
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.save_model:
                            filename = self.args.save_model + 'pretrain/' + myio.create_fname_identifier(self.args)
                            self.save_model(filename, self.args, pretrain=True)
                            json_train['BEST_DEV_EPOCH'] = epoch

                            myio.save_dev_results(self.args, None, dev_z, x, sha_ls)

            if more_count > 5:
                json_train['ERROR'] = 'Stuck reducing error rate, at epoch ' + str(epoch + 1) + '. LR = ' + str(lr_val)
                json.dump(json_train, ofp_train)
                ofp_train.close()
                return

        if unchanged > 20:
            json_train['UNCHANGED'] = unchanged

        json.dump(json_train, ofp_train)
        ofp_train.close()

    def evaluate_pretrain_data(self, eval_func):
        tot_obj = 0.0
        N = 0

        dev_z = []
        x = []
        sha_ls = []

        num_files = self.args.num_files_dev

        for i in xrange(num_files):
            if args.pad_repeat:
                batches_x, _, _, batches_bm, batches_sha, batches_rx, batches_fw, batches_cs = myio.load_batches(
                    self.args.batch_dir + self.args.source + 'dev', i)
            else:
                batches_x, _, _, batches_bm, _, batches_sha, batches_rx, batches_fw, batches_cs = myio.load_batches(
                    self.args.batch_dir + self.args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                bx, bm, sha, rx, bfw, bcs = batches_x[j], batches_bm[j], batches_sha[j], batches_rx[j], batches_fw[j], batches_cs[j]
                bz, l, o = eval_func(bx, bm, bfw, bcs)
                tot_obj += o
                N += len(bx)

                x.append(rx)
                dev_z.append(bz)
                sha_ls.append(sha)

        return tot_obj / float(N), dev_z, x, sha_ls

    def evaluate_pretrain_data_rouge(self, eval_func):
        tot_obj = 0.0
        N = 0

        dev_z = []
        x = []
        sha_ls = []

        num_files = self.args.num_files_dev

        for i in xrange(num_files):
            batches_x, _, _, batches_bm, batches_sha, batches_rx, batches_fw, batches_cs = myio.load_batches(
                self.args.batch_dir + self.args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                bx, bm, sha, rx, bfw, bcs = batches_x[j], batches_bm[j], batches_sha[j], batches_rx[j], batches_fw[j], batches_cs[j]
                bz, l, o = eval_func(bx, bm, bfw, bcs)
                tot_obj += o

                x.append(rx)
                dev_z.append(bz)
                sha_ls.append(sha)

            N += len(batches_x)

        myio.save_dev_results(self.args, None, dev_z, x, sha_ls)

        return tot_obj / float(N), dev_z

    def evaluate_data(self, eval_func):
        tot_obj = 0.0
        N = 0

        dev_z = []
        x = []
        sha_ls = []
        dev_acc = []

        num_files = self.args.num_files_dev

        for i in xrange(num_files):
            if args.pad_repeat:
                batches_x, batches_y, batches_e, batches_bm, batches_sha, batches_rx, batches_fw, batches_csz = myio.load_batches(
                    self.args.batch_dir + self.args.source + 'dev', i)
            else:
                batches_x, batches_y, batches_e, batches_bm, batches_lm,  batches_sha, batches_rx, batches_fw, batches_csz = myio.load_batches(
                    self.args.batch_dir + self.args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                if args.pad_repeat:
                    bx, by, be, bm, sha, rx, fw, csz = batches_x[j], batches_y[j], batches_e[j], batches_bm[j], \
                                                       batches_sha[j], batches_rx[j], batches_fw[j], batches_csz[j]
                    bz, o, e, preds = eval_func(bx, by, bm, be, fw, csz)
                else:
                    bx, by, be, bm, sha, rx, ble, fw, csz = batches_x[j], batches_y[j], batches_e[j], batches_bm[j], \
                                                            batches_sha[j], batches_rx[j], batches_lm[j], batches_fw[j], \
                                                            batches_csz[j]
                    bz, o, e, preds = eval_func(bx, by, bm, be, fw, csz, ble)

                tot_obj += o

                x.append(rx)
                dev_z.append(bz)
                sha_ls.append(sha)
                dev_acc.append(self.eval_acc(be, preds))

            N += cur_len

        return tot_obj / float(N), dev_z, x, sha_ls, np.mean(dev_acc)

    def evaluate_test_data(self, eval_func):
        N = 0

        test_z = []
        x = []
        y = []
        e =[]
        sha_ls = []

        num_files = self.args.num_files_test

        for i in xrange(num_files):
            batches_x, batches_y, batches_e, _, batches_sha, batches_rx, batches_fw, batches_csz = myio.load_batches(
                self.args.batch_dir + self.args.source + 'test', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                bx, rx, bsha, by, be, fw, csz = batches_x[j], batches_rx[j], batches_sha[j], batches_y[j], batches_e[j], batches_fw[j], batches_csz[j]
                bz = eval_func(bx, fw, csz)

                x.append(rx)
                y.append(by)
                e.append(be)

                test_z.append(bz)
                sha_ls.append(bsha)

            N += len(batches_x)

        return test_z, x, y, e, sha_ls

    def eval_acc(self,e, preds):
        gs = np.argmax(e, axis=1)
        system = np.argmax(preds, axis=1)

        return sk.accuracy_score(gs, system)


def main():
    assert args.embedding, "Pre-trained word embeddings required."

    vocab, parse_v = myio.get_vocab(args)
    embedding_layer = myio.create_embedding_layer(args, args.embedding, vocab, '<unk>')

    n_classes =args.nclasses

    model = Model(
        args=args,
        embedding_layer=embedding_layer,
        nclasses=n_classes
    )

    if args.dev_baseline:
        num_files = args.num_files_dev

        rx_ls = []
        bm_ls = []

        for i in xrange(num_files):
            batches_x, _, _, batches_bm, _, batches_sha, batches_rx = myio.load_batches(
                args.batch_dir + args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                _, bm, _, rx = batches_x[j], batches_bm[j], batches_sha[j], batches_rx[j]
                rx_ls.append(rx)
                bm_ls.append(bm)

        myio.eval_baseline(args, bm_ls, rx_ls, 'dev')
    elif args.test_baseline:
        num_files = args.num_files_test

        rx_ls = []
        bm_ls = []

        for i in xrange(num_files):
            batches_x, batches_bm, batches_sha, batches_rx = myio.load_batches(
                args.batch_dir + args.source + 'test', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                _, bm, _, rx = batches_x[j], batches_bm[j], batches_sha[j], batches_rx[j]
                rx_ls.append(rx)
                bm_ls.append(bm)

        myio.eval_baseline(args, bm_ls, rx_ls, 'test')

    elif args.train:

        if args.pretrain:
            model.ready_pretrain()
            model.pretrain()
        else:
            if args.load_model_pretrain:
                model.load_model_pretrain(args.save_model + 'pretrain/' + args.load_model)
            else:
                model.ready()

            model.train()

    elif args.dev:
        if args.pretrain:
            model.load_model_pretrain(args.save_model + 'pretrain/' + args.load_model)
            model.dev()
        else:
            model.load_model(args.save_model + args.load_model)
            model.dev_full()

    elif args.test:
        model.load_model(args.save_model + args.load_model, True)
        model.test()


if __name__ == "__main__":
    args = summarization_args.get_args()
    main()
