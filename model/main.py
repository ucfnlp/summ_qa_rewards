import cPickle as pickle
import gzip
import json
import os
import time

import numpy as np
import theano
import theano.tensor as T

import myio
import summarization_args
from nn.basic import LSTM, apply_dropout, Layer
from nn.extended_layers import ExtRCNN, ZLayer, ExtLSTM, HLLSTM
from nn.initialization import get_activation_by_name, softmax
from nn.optimization import create_optimization_updates
from util import say


class Generator(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # len*batch
        x = self.x = T.imatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]

        for i in xrange(2):
            l = LSTM(
                    n_in = n_e,
                    n_out = n_d,
                    activation = activation
                )
            layers.append(l)

        # len * batch
        masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)
        self.word_embs = embs

        flipped_embs = embs[::-1]

        # len*bacth*n_d
        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)
        h_final = T.concatenate([h1, h2[::-1]], axis=2)
        self.h_final = h_final = apply_dropout(h_final, dropout)
        size = n_d * 2

        output_layer = self.output_layer = ZLayer(
                n_in = size,
                n_hidden = args.hidden_dimension2,
                activation = activation
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
        y = self.y = T.imatrix('y')
        gold_standard_entities = self.gold_standard_entities = T.imatrix('gs')
        ve = self.ve = T.imatrix('ve')
        bm = self.bm = T.imatrix('bm')

        dropout = generator.dropout

        # len*batch
        x = generator.x
        z = generator.z_pred

        mask_x = T.cast(T.neq(x, padding_id) * z, theano.config.floatX)
        tiled_x_mask = T.tile(mask_x.dimshuffle(0,1,'x'), (args.n, 1)).dimshuffle((1, 0, 2))
        mask_y = T.cast(T.neq(y, padding_id), theano.config.floatX).dimshuffle((0, 1, 'x'))

        softmax_mask = T.zeros_like(tiled_x_mask) - 1e8
        softmax_mask = softmax_mask * (tiled_x_mask - 1) * -1

        # Duplicate both x, and valid entity masks
        gen_h_final = T.tile(generator.h_final, (args.n, 1)).dimshuffle((1, 0, 2))
        ve_tiled = T.tile(ve, (args.n, 1))

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

        embs = embedding_layer.forward(y.ravel())
        embs = embs.reshape((y.shape[0], y.shape[1], n_e))

        flipped_embs = embs[::-1]
        flipped_mask = mask_y[::-1]

        h_f = rnn_fw.forward_all(embs, mask_y)
        h_r = rnn_rv.forward_all(flipped_embs, flipped_mask)

        h_concat = T.concatenate([h_f, h_r], axis=2).dimshuffle((1, 2, 0))

        inp_dot_hl = T.batched_dot(gen_h_final, h_concat)
        inp_dot_hl = inp_dot_hl - softmax_mask
        inp_dot_hl = inp_dot_hl.ravel()

        alpha = T.nnet.softmax(inp_dot_hl.reshape((args.n * args.batch, args.inp_len)))

        o = T.batched_dot(alpha, gen_h_final)

        output_layer = Layer(
            n_in=n_d*2,
            n_out=self.nclasses,
            activation=softmax
        )

        layers.append(rnn_fw)
        layers.append(rnn_rv)
        layers.append(output_layer)

        preds = output_layer.forward(o) * ve_tiled
        preds_clipped = T.clip(preds, 1e-7, 1.0 - 1e-7)
        cross_entropy = T.nnet.categorical_crossentropy(preds_clipped, gold_standard_entities)
        loss_mat = cross_entropy.reshape((args.batch, args.n))

        padded = T.shape_padaxis(T.zeros_like(z[0]), axis=1).dimshuffle((1, 0))
        z_shift = T.concatenate([z[1:], padded], axis=0)

        valid_bg = z * z_shift
        bigram_ol = valid_bg * bm

        total_z_bg_per_sample = T.sum(bigram_ol, axis=0)
        total_bg_per_sample = T.sum(bm, axis=0) + args.bigram_smoothing
        self.bigram_loss = bigram_loss = total_z_bg_per_sample / total_bg_per_sample

        self.loss_vec = loss_vec = T.mean(loss_mat, axis=1)

        zsum = generator.zsum
        zdiff = generator.zdiff
        logpz = generator.logpz

        loss = self.loss = T.mean(loss_vec)
        self.zsum = zsum = T.abs_(zsum / T.sum(T.neq(x, padding_id), axis=0, dtype=theano.config.floatX) - 0.15)

        cost_vec = loss_vec + args.sparsity*(zsum - bigram_loss) + zdiff * args.coherent
        baseline = T.mean(cost_vec)
        cost_vec = cost_vec - baseline

        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))
        self.obj = T.mean(cost_vec)

        params = self.params = []
        for l in layers:
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

        self.cost_g = cost_logpz * 10 + generator.l2_cost
        self.cost_e = loss * 10 + l2_cost

    # def masked_softmax(self, a, m, axis=0):
    #     e_a = T.exp(a)
    #     masked_e = e_a * m
    #     sum_masked_e = T.sum(masked_e, axis, keepdims=True)
    #     return masked_e / sum_masked_e

class Model(object):
    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
        self.generator = Generator(args, embedding_layer, nclasses)
        self.encoder = Encoder(args, nclasses, self.generator)
        self.generator.ready()
        self.encoder.ready()
        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.y = self.encoder.y
        self.bm = self.encoder.bm
        self.gold_standard_entities = self.encoder.gold_standard_entities
        self.ve = self.encoder.ve
        self.z = self.generator.z_pred
        self.params = self.encoder.params + self.generator.params

    def evaluate_rnn_weights(self, args, e, b):
        fout= gzip.open(args.weight_eval + 'e_' + str(e) + '_b_' + str(b) + '_weights.pkl.gz', 'wb+')

        pickle.dump(
            ([x.get_value() for x in self.encoder.encoder_params]),
            fout,
            protocol=pickle.HIGHEST_PROTOCOL
        )

        fout.close()

    def save_model(self, path, args):
        # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        # output to path
        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([x.get_value() for x in self.encoder.params],  # encoder
                 [x.get_value() for x in self.generator.params],  # generator
                 self.nclasses,
                 args  # training configuration
                 ),
                fout,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
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
        self.ready()
        for x, v in zip(self.encoder.params, eparams):
            x.set_value(v)
        for x, v in zip(self.generator.params, gparams):
            x.set_value(v)

    def train(self, train, dev, test, rationale_data):
        args = self.args
        dropout = self.dropout
        padding_id = self.embedding_layer.vocab_map["<padding>"]

        if dev is not None:
            dev_batches_x, dev_batches_y, dev_batches_ve, dev_batches_e, dev_batches_bm, dev_batches_cy = myio.create_batches(
                args, self.nclasses, dev[0], dev[1], dev[2], dev[3], dev[4], args.batch,  padding_id
            )

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
            inputs=[self.x, self.y, self.bm, self.gold_standard_entities, self.ve],
            outputs=[self.z, self.encoder.obj, self.encoder.loss],
            updates=self.generator.sample_updates
        )

        train_generator = theano.function(
            inputs=[self.x, self.y, self.bm, self.gold_standard_entities, self.ve],
            outputs=[self.encoder.obj, self.encoder.loss, self.z, self.encoder.zsum, self.encoder.bigram_loss, self.encoder.loss_vec, self.generator.zdiff],
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

        for epoch in xrange(args.max_epochs):
            total_words_per_epoch = 0
            total_summaries_per_epoch = 0
            unchanged += 1
            more_count = 0

            if unchanged > 20:
                break

            train_batches_x, train_batches_y, train_batches_ve, train_batches_e, train_batches_bm, _ = myio.create_batches(
                args, self.nclasses, train[0], train[1], train[2], train[3], None, args.batch, padding_id
            )

            more = True
            if args.decay_lr:
                param_bak = [p.get_value(borrow=False) for p in self.params]

            while more:
                processed = 0
                train_cost = 0.0
                train_loss = 0.0
                p1 = 0.0
                more_count += 1

                if more_count > 10:
                    break
                start_time = time.time()

                loss_all = []
                obj_all = []
                zsum_all = []
                bigram_loss_all = []
                loss_vec_all = []
                z_diff_all = []

                N = len(train_batches_x)
                print N, 'batches'
                for i in xrange(N):
                    if (i + 1) % 100 == 0:
                        say("\r{}/{} {:.2f}       ".format(i + 1, N, p1 / (i + 1)))

                    bx, by, bve, be, bm = train_batches_x[i], train_batches_y[i], train_batches_ve[i], train_batches_e[i], train_batches_bm[i]
                    mask = bx != padding_id

                    if len(bx[0]) != args.batch:
                        print 'B_len', len(bx[0])
                        break

                    cost, loss, z, zsum, bigram_loss, loss_vec, zdiff = train_generator(bx, by, bm, be, bve)
                    obj_all.append(cost)
                    loss_all.append(loss)
                    zsum_all.append(zsum)
                    bigram_loss_all.append(bigram_loss)
                    loss_vec_all.append(loss_vec)
                    z_diff_all.append(zdiff)

                    k = len(by)
                    processed += k
                    train_cost += cost
                    train_loss += loss
                    p1 += np.sum(z * mask) / (np.sum(mask) + 1e-8)

                    total_summaries_per_epoch += args.batch
                    total_words_per_epoch += myio.total_words(z)

                cur_train_avg_cost = train_cost / N

                if dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_z = self.evaluate_data(dev_batches_x, dev_batches_y, dev_batches_ve, dev_batches_e,
                                                        dev_batches_bm, eval_generator)
                    self.dropout.set_value(dropout_prob)
                    cur_dev_avg_cost = dev_obj

                    myio.save_dev_results(args, epoch, dev_obj, dev_z, dev_batches_x,  dev_batches_cy,  self.embedding_layer)
                more = False

                if args.decay_lr and last_train_avg_cost is not None:
                    if cur_train_avg_cost > last_train_avg_cost * (1 + tolerance):
                        more = True
                        say("\nTrain cost {} --> {}\n".format(
                            last_train_avg_cost, cur_train_avg_cost
                        ))
                    if dev and cur_dev_avg_cost > last_dev_avg_cost * (1 + tolerance):
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
                    (time.time() - start_time) / 60.0 / (i + 1) * N
                ))

                if dev:
                    last_dev_avg_cost = cur_dev_avg_cost
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.save_model:
                            self.save_model(args.save_model, args)

            if more_count > 10:
                json_train['ERROR'] = 'Stuck reducing error rate, at epoch ' + str(epoch + 1) + '. LR = ' + str(lr_val)
                json.dump(json_train, ofp_train)
                ofp_train.close()
                return

        if unchanged > 0:
            json_train['UNCHANGED'] = unchanged

        json.dump(json_train, ofp_train)
        ofp_train.close()

    def evaluate_data(self, batches_x, batches_y, batches_ve, batches_e, batches_bm, eval_func):
        tot_obj = 0.0
        dev_z = []

        for bx, by, bve, be, bm in zip(batches_x, batches_y, batches_ve, batches_e, batches_bm):
            bz, o, e = eval_func(bx, by, bm, be, bve)
            tot_obj += o

            dev_z.append(bz)

        n = float(len(batches_x))

        return tot_obj / n, dev_z


def main():
    assert args.embedding, "Pre-trained word embeddings required."

    vocab = myio.get_vocab(args)
    embedding_layer = myio.create_embedding_layer(args, args.embedding, vocab)

    entities = myio.load_e(args)
    n_classes = len(entities)

    if args.train:
        train_x, train_y, train_e_idxs, train_e = myio.read_docs(args, 'train')

    if args.dev:
        dev_x, dev_y, dev_e_idxs, dev_e, dev_cy = myio.read_docs(args, 'dev')

    if args.train:
        model = Model(
            args=args,
            embedding_layer=embedding_layer,
            nclasses=n_classes
        )
        model.ready()

        model.train(
            (train_x, train_y, train_e_idxs, train_e),
            (dev_x, dev_y, dev_e_idxs, dev_e, dev_cy),
            None,  # (test_x, test_y),
            None,
        )


if __name__ == "__main__":
    print theano.config.exception_verbosity
    args = summarization_args.get_args()
    main()
