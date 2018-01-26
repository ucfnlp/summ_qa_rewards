import cPickle as pickle
import gzip
import json
import os
import time
import random

import numpy as np
import theano
import theano.tensor as T

import myio
import summarization_args
from nn.basic import LSTM, apply_dropout, Layer
from nn.extended_layers import ExtRCNN, ZLayer, ExtLSTM, HLLSTM, RCNN
from nn.initialization import get_activation_by_name, softmax
from nn.optimization import create_optimization_updates
from util import say


class Generator(object):

    def __init__(self, args, embedding_layer):
        self.args = args
        self.embedding_layer = embedding_layer

    def ready(self):
        embedding_layer = self.embedding_layer
        args = self.args
        self.padding_id = padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # inp_len x batch
        x = self.x = T.imatrix('x')

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]

        for i in xrange(2):
            if args.layer == 'lstm':
                l = LSTM(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation,
                )
            else:
                l = RCNN(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation,
                    order=args.order
                )
            layers.append(l)

        self.masks = masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

        embs = embedding_layer.forward(x.ravel())

        self.word_embs = embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)

        flipped_embs = embs[::-1]

        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)
        h_final = T.concatenate([h1, h2[::-1]], axis=2)
        self.h_final = h_final = apply_dropout(h_final, dropout)
        size = n_d * 2

        output_layer = self.output_layer = ZLayer(
                n_in = size,
                n_hidden = args.hidden_dimension2,
                activation = activation,
                layer=args.layer,
                test = (len(args.test) > 0)
            )

        z_pred, sample_updates = output_layer.sample_all(h_final)

        z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
        self.sample_updates = sample_updates

        probs = output_layer.forward_all(h_final, z_pred)

        logpz = - T.nnet.binary_crossentropy(probs, z_pred) * masks
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape)

        # batch
        z = z_pred
        self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
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

        z_shift = self.z_pred[1:]
        z_new = self.z_pred[:-1]

        valid_bg = z_new * z_shift
        bigram_ol = valid_bg * bm[:-1]

        total_z_bg_per_sample = T.sum(bigram_ol, axis=0)
        total_bg_per_sample = T.sum(bm, axis=0) + args.bigram_smoothing
        self.bigram_loss = bigram_loss = total_z_bg_per_sample / total_bg_per_sample

        z_totals = T.sum(self.masks, axis=0, dtype=theano.config.floatX)

        zsum = T.abs_(self.zsum / z_totals - 0.15)
        zdiff = self.zdiff / z_totals

        self.logpz = logpz = T.sum(self.logpz, axis=0)

        self.cost_vec = cost_vec = (1 - bigram_loss) + args.coeff_summ_len * zsum + args.coeff_fluency * zdiff

        self.cost_logpz = cost_logpz = T.mean(cost_vec * logpz)

        self.obj = T.mean(cost_vec)
        self.cost_g = cost_logpz * args.coeff_cost_scale + self.l2_cost


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

        # hl_inp_len x (batch * n)
        y = self.y = T.imatrix('y')
        # (batch * n) x n_classes
        gold_standard_entities = self.gold_standard_entities = T.imatrix('gs')
        # inp_len x batch
        bm = self.bm = T.imatrix('bm')

        # inp_len x batch
        x = generator.x
        z = generator.z_pred

        mask_x = T.cast(T.neq(x, padding_id) * z, theano.config.floatX).dimshuffle(0,1,'x')
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

        # (batch * n) x inp_len x 1
        inp_dot_hl = T.batched_dot(gen_h_final, h_concat_y)
        inp_dot_hl = inp_dot_hl - softmax_mask
        inp_dot_hl = inp_dot_hl.ravel()

        alpha = T.nnet.softmax(inp_dot_hl.reshape((args.n * x.shape[1], args.inp_len)))
        # inp_dot_hl = T.batched_dot(gen_h_final, h_concat_y)
        # inp_dot_hl = inp_dot_hl.ravel()
        # tiled_x_mask = tiled_x_mask.ravel()

        # alpha = self.softmax_mask(inp_dot_hl.reshape((args.n * x.shape[1], args.inp_len)),
        #                           tiled_x_mask.reshape((args.n * x.shape[1], args.inp_len)))

        o = T.batched_dot(alpha, gen_h_final)

        output_layer = Layer(
            n_in=n_d*2,
            n_out=self.nclasses,
            activation=softmax
        )

        layers.append(rnn_fw)
        layers.append(rnn_rv)
        layers.append(output_layer)

        preds = output_layer.forward(o)
        preds_clipped = T.clip(preds, 1e-7, 1.0 - 1e-7)
        cross_entropy = T.nnet.categorical_crossentropy(preds_clipped, gold_standard_entities)
        loss_mat = cross_entropy.reshape((x.shape[1], args.n))

        z_shift = z[1:]
        z_new = z[:-1]

        valid_bg = z_new * z_shift
        bigram_ol = valid_bg * bm[:-1]

        total_z_bg_per_sample = T.sum(bigram_ol, axis=0)
        total_bg_per_sample = T.sum(bm, axis=0) + args.bigram_smoothing
        self.bigram_loss = bigram_loss = total_z_bg_per_sample / total_bg_per_sample

        self.loss_vec = loss_vec = T.mean(loss_mat, axis=1)

        self.zsum = zsum = generator.zsum
        self.zdiff = zdiff = generator.zdiff
        logpz = generator.logpz

        loss = self.loss = T.mean(loss_vec)

        z_totals = T.sum(T.neq(x, padding_id), axis=0, dtype=theano.config.floatX)
        self.zsum = zsum = T.abs_(self.zsum / z_totals - 0.15)
        self.zdiff = zdiff = zdiff / z_totals

        cost_vec = loss_vec + args.coeff_adequacy * (
                (1 - bigram_loss) + args.coeff_summ_len * zsum + args.coeff_fluencys * zdiff)

        baseline = T.mean(cost_vec)
        self.cost_vec = cost_vec = cost_vec - baseline

        self.logpz = logpz = T.sum(logpz, axis=0)
        self.cost_logpz = cost_logpz = T.mean(cost_vec * logpz)
        self.obj = T.mean(cost_vec)

        params = self.params = []
        for l in layers:
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
        self.cost_e = loss * args.coeff_cost_scale + l2_cost

    # def softmax_mask(self, x, mask):
    #     x = T.nnet.softmax(x)
    #     x = x * mask
    #     x = x / x.sum(0, keepdims=True)
    #     return x


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
        self.y = self.encoder.y
        self.bm = self.encoder.bm
        self.gold_standard_entities = self.encoder.gold_standard_entities
        self.z = self.generator.z_pred
        self.params = self.encoder.params + self.generator.params

    def ready_test(self):
        args, embedding_layer= self.args, self.embedding_layer
        self.generator = Generator(args, embedding_layer)
        self.generator.ready()
        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.z = self.generator.z_pred

    def ready_pretrain(self):
        args, embedding_layer= self.args, self.embedding_layer
        self.generator = Generator(args, embedding_layer)
        self.generator.ready()
        self.generator.pretrain()

        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.bm = self.generator.bm
        self.z = self.generator.z_pred
        self.params = self.generator.params

    def evaluate_rnn_weights(self, args, e, b):
        fout= gzip.open(args.weight_eval + 'e_' + str(e) + '_b_' + str(b) + '_weights.pkl.gz', 'wb+')

        pickle.dump(
            ([x.get_value() for x in self.encoder.encoder_params]),
            fout,
            protocol=pickle.HIGHEST_PROTOCOL
        )

        fout.close()

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

    def test(self, test):
        args = self.args

        test_generator = theano.function(
            inputs=[self.x],
            outputs=self.z,
            updates=self.generator.sample_updates
        )

        padding_id = self.embedding_layer.vocab_map["<padding>"]
        test_batches_x = myio.create_test(args, test, padding_id)

        self.dropout.set_value(0.0)
        z = self.evaluate_test_data(test_batches_x, test_generator)

        myio.save_test_results_rouge(args, z, test_batches_x, self.embedding_layer)

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

        eval_generator = theano.function(
            inputs=[self.x, self.y, self.bm, self.gold_standard_entities],
            outputs=[self.z, self.encoder.obj, self.encoder.loss],
            updates=self.generator.sample_updates
        )

        train_generator = theano.function(
            inputs=[self.x, self.y, self.bm, self.gold_standard_entities],
            outputs=[self.encoder.obj, self.encoder.loss, self.z, self.encoder.zsum, self.encoder.zdiff,
                     self.encoder.bigram_loss, self.encoder.loss_vec, self.encoder.cost_logpz, self.encoder.logpz, self.generator.probs, self.generator.z_pred, self.encoder.cost_vec, self.generator.masks],
            updates=updates_e.items() + updates_g.items() + self.generator.sample_updates
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

        rouge_fname = None

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

                loss_all = []
                obj_all = []
                zsum_all = []
                bigram_loss_all = []
                loss_vec_all = []
                z_diff_all = []
                cost_logpz_all = []
                logpz_all = []
                probs_all = []
                z_pred_all = []
                cost_vec_all = []

                num_files = args.num_files_train
                N = args.online_batch_size * num_files

                for i in xrange(num_files):
                    train_batches_x, train_batches_y, train_batches_e, train_batches_bm = myio.load_batches(
                        args.batch_dir + args.source + 'train', i)

                    random.seed(5817)
                    perm2 = range(len(train_batches_x))
                    random.shuffle(perm2)

                    train_batches_x = [train_batches_x[k] for k in perm2]
                    train_batches_y = [train_batches_y[k] for k in perm2]
                    train_batches_e = [train_batches_e[k] for k in perm2]
                    train_batches_bm = [train_batches_bm[k] for k in perm2]

                    cur_len = len(train_batches_x)

                    for j in xrange(cur_len):
                        if args.full_test:
                            if (i* args.online_batch_size + j + 1) % 100 == 0:
                                say("\r{}/{} {:.2f}       ".format(i* args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))
                        elif (i* args.online_batch_size + j + 1) % 10 == 0:
                                say("\r{}/{} {:.2f}       ".format(i* args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))

                        bx, by, be, bm = train_batches_x[j], train_batches_y[j], train_batches_e[j], train_batches_bm[j]
                        mask = bx != padding_id

                        cost, loss, z, zsum, zdiff, bigram_loss, loss_vec, cost_logpz, logpz, g_probs, g_z_pred, cost_vec, masks = train_generator(
                            bx, by, bm, be)
                        obj_all.append(cost)
                        loss_all.append(loss)
                        zsum_all.append(zsum)
                        bigram_loss_all.append(bigram_loss)
                        loss_vec_all.append(loss_vec)
                        z_diff_all.append(zdiff)
                        cost_logpz_all.append(cost_logpz)
                        logpz_all.append(logpz)
                        probs_all.append(g_probs)
                        z_pred_all.append(z)
                        cost_vec_all.append(cost_vec)

                        train_cost += cost
                        train_loss += loss

                        # print 'cost_logpz, logpz, g_probs',cost_logpz, logpz,g_probs.shape, g_z_pred.shape
                        print 'sum(z)', np.sum(z)
                        # print masks
                        p1 += np.sum(z * mask) / (np.sum(mask) + 1e-8)

                cur_train_avg_cost = train_cost / N

                if args.dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_z, rouge_fname = self.evaluate_data(eval_generator)
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

                if args.sanity_check:
                    myio.record_observations_verbose(json_train, epoch + 1, loss_all, obj_all, zsum_all, loss_vec_all,
                                             z_diff_all, cost_logpz_all, logpz_all, probs_all, z_pred_all, cost_vec_all)
                else:
                    myio.record_observations(json_train, epoch + 1, loss_all, obj_all, zsum_all, bigram_loss_all,
                                             loss_vec_all, z_diff_all)

                last_train_avg_cost = cur_train_avg_cost

                say("\n")
                say(("Generator Epoch {:.2f}  costg={:.4f}  lossg={:.4f}  " +
                     "\t[{:.2f}m / {:.2f}m]\n").format(
                    epoch + (i + 1.0) / N,
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

            if more_count > 5:
                json_train['ERROR'] = 'Stuck reducing error rate, at epoch ' + str(epoch + 1) + '. LR = ' + str(lr_val)
                json.dump(json_train, ofp_train)
                ofp_train.close()

                myio.get_rouge(args, rouge_fname)
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
            inputs=[self.x, self.bm],
            outputs=[self.z, self.generator.cost_g, self.generator.obj],
            updates=self.generator.sample_updates
        )

        train_generator = theano.function(
            inputs=[self.x, self.bm],
            outputs=[self.generator.obj, self.z, self.generator.zsum, self.generator.zdiff, self.generator.bigram_loss, self.generator.cost_g, self.generator.logpz],
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
        rouge_fname = None

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
                bigram_loss_all = []
                z_diff_all = []
                z_pred_all = []

                num_files = args.num_files_train
                N = args.online_batch_size * num_files

                for i in xrange(num_files):
                    train_batches_x, train_batches_y, train_batches_e, train_batches_bm = myio.load_batches(
                        args.batch_dir + args.source + 'train', i)

                    random.seed(5817)
                    perm2 = range(len(train_batches_x))
                    random.shuffle(perm2)

                    train_batches_x = [train_batches_x[k] for k in perm2]
                    train_batches_bm = [train_batches_bm[k] for k in perm2]
                    cur_len = len(train_batches_x)

                    for j in xrange(cur_len):
                        if args.full_test:
                            if (i * args.online_batch_size + j + 1) % 10 == 0:
                                say("\r{}/{} {:.2f}       ".format(i * args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))
                        elif (i * args.online_batch_size + j + 1) % 10 == 0:
                            say("\r{}/{} {:.2f}       ".format(i * args.online_batch_size + j + 1, N, p1 / (i * args.online_batch_size + j + 1)))

                        bx, bm = train_batches_x[j], train_batches_bm[j]
                        mask = bx != padding_id

                        obj, z, zsum, zdiff, bigram_loss,cost_g, logpz = train_generator(bx, bm)
                        zsum_all.append(zsum)
                        bigram_loss_all.append(bigram_loss)
                        z_diff_all.append(zdiff)
                        z_pred_all.append(z)
                        obj_all.append(obj)

                        train_cost += obj

                        p1 += np.sum(z * mask) / (np.sum(mask) + 1e-8)

                cur_train_avg_cost = train_cost / N

                if args.dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_z, rouge_fname = self.evaluate_pretrain_data(eval_generator)
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
                    epoch + (i + 1.0) / N,
                    train_cost / N,
                    train_loss / N,
                    (time.time() - start_time) / 60.0,
                    (time.time() - start_time) / 60.0 / (i + 1) * N
                ))

                if args.dev:
                    last_dev_avg_cost = cur_dev_avg_cost
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.save_model:
                            filename = args.save_model + 'pretrain/' + myio.create_fname_identifier(args)
                            self.save_model(filename, args, pretrain=True)

            if more_count > 5:
                json_train['ERROR'] = 'Stuck reducing error rate, at epoch ' + str(epoch + 1) + '. LR = ' + str(lr_val)
                json.dump(json_train, ofp_train)
                ofp_train.close()

                myio.get_rouge(args, rouge_fname)
                return

        if unchanged > 20:
            json_train['UNCHANGED'] = unchanged

        myio.get_rouge(args, rouge_fname)

        json.dump(json_train, ofp_train)
        ofp_train.close()

    def evaluate_pretrain_data(self, eval_func):
        tot_obj = 0.0
        dev_z = []
        x = []
        num_files = args.num_files_dev
        N = 0

        for i in xrange(num_files):
            batches_x, _, _, batches_bm = myio.load_batches(
                args.batch_dir + args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                for bx, bm in zip(batches_x, batches_bm):
                    bz, l, o = eval_func(bx, bm)
                    tot_obj += o
                    N += len(bx)
                    x.append(bx)
                    dev_z.append(bz)

        rouge_fname = myio.save_dev_results(args, None, dev_z, x, self.embedding_layer, pretrain=True)

        return tot_obj / float(N), dev_z, rouge_fname

    def evaluate_data(self, eval_func):
        tot_obj = 0.0
        dev_z = []
        x = []
        num_files = args.num_files_dev
        N = 0

        for i in xrange(num_files):
            batches_x, batches_y, batches_e, batches_bm = myio.load_batches(
                args.batch_dir + args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                for bx, by, be, bm in zip(batches_x, batches_y, batches_e, batches_bm):
                    bz, o, e = eval_func(bx, by, bm, be)
                    tot_obj += o
                    N += len(bx)
                    x.append(bx)
                    dev_z.append(bz)

        rouge_fname = myio.save_dev_results(args, None, dev_z, x, self.embedding_layer, pretrain=True)

        return tot_obj / float(N), dev_z, rouge_fname

    def evaluate_test_data(self, batches_x, eval_func):
        dev_z = []

        for bx in zip(batches_x):
            bz = eval_func(bx)

            dev_z.append(bz)

        return dev_z


def main():
    assert args.embedding, "Pre-trained word embeddings required."

    vocab = myio.get_vocab(args)
    embedding_layer = myio.create_embedding_layer(args, args.embedding, vocab)

    entities = myio.load_e(args)
    n_classes = len(entities)

    if args.test:
        test_x = myio.read_docs(args, 'test')

    model = Model(
        args=args,
        embedding_layer=embedding_layer,
        nclasses=n_classes
    )

    if args.batch_data:
        if args.train:
            train_x, train_y, train_e = myio.read_docs(args, 'train')
            train_batches_x, train_batches_y, train_batches_e, train_batches_bm = myio.create_batches(args, model.nclasses, train_x, train_y, train_e, args.batch, embedding_layer.vocab_map["<padding>"])
            myio.save_batched(args, train_batches_x, train_batches_y, train_batches_e, train_batches_bm, 'train')

        if args.dev:
            dev_x, dev_y, dev_e = myio.read_docs(args, 'dev')
            dev_batches_x, dev_batches_y, dev_batches_e, dev_batches_bm = myio.create_batches(args, model.nclasses, dev_x, dev_y, dev_e, args.batch,
                                embedding_layer.vocab_map["<padding>"], sort=False)

            myio.save_batched(args, dev_batches_x, dev_batches_y, dev_batches_e, dev_batches_bm, 'dev')

    elif args.train:

        if args.pretrain:
            model.ready_pretrain()
            model.pretrain()
        else:
            model.ready()
            model.train()

    elif args.test:
        model.load_model(args.save_model + args.load_file, True)

        model.test(test_x)


if __name__ == "__main__":
    args = summarization_args.get_args()
    main()
