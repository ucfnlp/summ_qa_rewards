import cPickle as pickle
import gzip
import json
import os
import time

import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import jaccard_similarity_score

import myio
import summarization_args
from nn.advanced import RCNN
from nn.basic import LSTM, apply_dropout, Layer
from nn.extended_layers import ExtRCNN, ZLayer, ExtLSTM, HLLSTM
from nn.initialization import get_activation_by_name
from nn.optimization import create_optimization_updates
from util import say, get_ngram


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

        # sample z given text (i.e. x)
        z_pred, sample_updates = output_layer.sample_all(h_final)

        # we are computing approximated gradient by sampling z;
        # so should mark sampled z not part of the gradient propagation path
        #
        z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
        self.sample_updates = sample_updates
        print "z_pred", z_pred.ndim

        probs = output_layer.forward_all(h_final, z_pred)
        print "probs", probs.ndim

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
        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                                        for x in params)
        say("total # parameters: {}\n".format(nparams))

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

        y = self.y = T.imatrix('y')
        gold_standard_entities = self.gold_standard_entities = T.imatrix('gs')
        ve = self.ve = T.imatrix('ve')

        dropout = generator.dropout

        # len*batch
        x = generator.x
        z = generator.z_pred

        mask_x = T.cast(T.neq(x, padding_id) * z, theano.config.floatX).dimshuffle((0,1,'x'))
        tiled_x_mask = T.tile(mask_x, (args.n, 1)).dimshuffle((1, 0))
        mask_y = T.cast(T.neq(y, padding_id), theano.config.floatX).dimshuffle((0, 1, 'x'))

        # Duplicate both x, and valid entity masks
        gen_h_final = T.tile(generator.h_final, (args.n, 1)).dimshuffle((1, 0, 2))
        ve = T.tile(ve, (args.n, 1)).dimshuffle((1, 0))

        n_d = args.hidden_dimension/2
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

        alpha = self.masked_softmax(inp_dot_hl.reshape((args.n * args.batch, args.inp_len)), tiled_x_mask)

        o = T.batched_dot(alpha, gen_h_final)

        output_layer = Layer(
            n_in=o.shape[1],
            n_out=args.nclasses,
            activation="softmax"
        )

        preds = output_layer.forward(o) * ve
        cross_entropy = T.nnet.categorical_crossentropy(preds, gold_standard_entities)
        loss_mat = cross_entropy.reshape((args.batch, args.n))

        self.loss_vec = loss_vec = T.mean(loss_mat, axis=1)

        zsum = generator.zsum
        zdiff = generator.zdiff
        logpz = generator.logpz

        coherent_factor = args.sparsity * args.coherent
        loss = self.loss = T.mean(loss_vec)
        sparsity_cost = self.sparsity_cost = T.mean(zsum) * args.sparsity + \
                                             T.mean(zdiff) * coherent_factor
        cost_vec = loss_vec + zsum * args.sparsity + zdiff * coherent_factor
        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))
        self.obj = T.mean(cost_vec)

        params = self.params = []
        for l in layers + [output_layer]:
            for p in l.params:
                params.append(p)
        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                      for x in params)
        say("total # parameters: {}\n".format(nparams))

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

    def masked_softmax(self, a, m, axis=0):
        e_a = T.exp(a)
        masked_e = e_a * m
        sum_masked_e = T.sum(masked_e, axis, keepdims=True)
        return masked_e / sum_masked_e

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
        self.bv = self.encoder.bv
        self.z = self.generator.z_pred_combined
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
        # (args, n_classes, x, y, ve, e, batch_size, padding_id, sort=True):
        if dev is not None:
            dev_batches_x, dev_batches_y, dev_batches_ve, dev_batches_e, dev_batches_bm = myio.create_batches(
                args, self.nclasses, dev[0], dev[1], dev[2], dev[3], args.batch,  padding_id
            )
        if test is not None:
            test_batches_x, test_batches_y = myio.create_batches(
                test[0], test[1], args.batch, padding_id
            )
        if rationale_data is not None:
            valid_batches_x, valid_batches_y = myio.create_batches(
                [u["xids"] for u in rationale_data],
                [u["y"] for u in rationale_data],
                args.batch,
                padding_id,
                sort=False
            )

        # start_time = time.time()
        # train_batches_x, train_batches_y = myio.create_batches(
        #     train[0], train[1], args.batch, padding_id
        # )
        # say("{:.2f}s to create training batches\n\n".format(
        #     time.time() - start_time
        # ))

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

        sample_generator = theano.function(
            inputs=[self.x],
            outputs=self.z,
            updates=self.generator.sample_updates
        )

        # get_loss_and_pred = theano.function(
        #     inputs=[self.x, self.y],
        #     outputs=[self.encoder.loss_vec, self.z],
        #     updates=self.generator.sample_updates + self.generator.sample_updates_sent
        # )
        #
        eval_generator = theano.function(
            inputs=[self.x, self.y, self.bv],
            outputs=[self.z, self.encoder.obj, self.encoder.loss],
            updates=self.generator.sample_updates
        )

        train_generator = theano.function(
            inputs=[self.x, self.y, self.bv],
            outputs=[self.encoder.obj, self.encoder.loss, self.encoder.sparsity_cost, self.z, gnorm_e, gnorm_g],
            updates=updates_e.items() + updates_g.items() + self.generator.sample_updates
        )

        eval_period = args.eval_period
        unchanged = 0
        best_dev = 1e+2
        best_dev_e = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.10 + 1e-3
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        metric_output = open(args.train_output_readable + '_METRICS' + '_sparcity_' + str(args.sparsity) + '.out', 'w+')

        if args.dev_baseline:
            ofp1 = open(args.train_output_readable + '_METRICS' + '_sparcity_' + str(args.sparsity) + '_baseline.out', 'w+')
            ofp2 = open(args.train_output_readable + '_sparcity_' + str(args.sparsity) + '_baseline.out', 'w+')

            dz = myio.convert_bv_to_z(dev_batches_bv)

            myio.write_train_results(dz[0], dev_batches_x[0], dev_batches_y[0], self.embedding_layer, ofp2, padding_id)
            myio.write_summ_for_rouge(args, dz, dev_batches_x, dev_batches_y, self.embedding_layer)
            myio.write_metrics(-1, -1, ofp1, -1, args)

            ofp1.close()
            ofp2.close()

        for epoch in xrange(args.max_epochs):
            read_output = open(args.train_output_readable + '_e_' + str(epoch) + '_sparcity_' + str(args.sparsity) + '.out', 'w+')
            total_words_per_epoch = 0
            total_summaries_per_epoch = 0
            unchanged += 1
            if unchanged > 20:
                metric_output.write("PROBLEM TRAINING, NO DEV IMPROVEMENT")
                metric_output.close()
                break

            train_batches_x, train_batches_y, train_batches_bv = myio.create_batches(
                train[0], train[1], args.batch, padding_id
            )

            more = True
            if args.decay_lr:
                param_bak = [p.get_value(borrow=False) for p in self.params]

            while more:
                processed = 0
                train_cost = 0.0
                train_loss = 0.0
                train_sparsity_cost = 0.0
                p1 = 0.0
                start_time = time.time()

                N = len(train_batches_x)
                for i in xrange(N):
                    if (i + 1) % 32 == 0:
                        say("\r{}/{} {:.2f}       ".format(i + 1, N, p1 / (i + 1)))

                    bx, by, bv = train_batches_x[i], train_batches_y[i], train_batches_bv[i]
                    mask = bx != padding_id

                    cost, loss, sparsity_cost, bz, gl2_e, gl2_g = train_generator(bx, by, bv)

                    if i % 64 == 0:
                        self.evaluate_rnn_weights(args, epoch, i)

                    if i % 8 == 0:
                        myio.write_train_results(bz, bx, by, self.embedding_layer, read_output, padding_id)

                    k = len(by)
                    processed += k
                    train_cost += cost
                    train_loss += loss
                    train_sparsity_cost += sparsity_cost
                    p1 += np.sum(bz * mask) / (np.sum(mask) + 1e-8)

                    total_summaries_per_epoch += args.batch
                    total_words_per_epoch += myio.total_words(bz)

                cur_train_avg_cost = train_cost / N

                if dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_loss, dev_p1, dev_v, dev_x, dev_y = self.evaluate_data(
                        dev_batches_x, dev_batches_y, dev_batches_bv, eval_generator, sampling=True)

                    self.dropout.set_value(dropout_prob)
                    cur_dev_avg_cost = dev_obj

                    myio.write_train_results(dev_v[0], dev_x[0], dev_y[0], self.embedding_layer, read_output, padding_id)
                    myio.write_summ_for_rouge(args, dev_v, dev_x, dev_y, self.embedding_layer)
                    myio.write_metrics(total_summaries_per_epoch, total_words_per_epoch, metric_output, epoch, args)

                    metric_output.flush()

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

                last_train_avg_cost = cur_train_avg_cost
                if dev: last_dev_avg_cost = cur_dev_avg_cost

                say("\n")
                say(("Generator Epoch {:.2f}  costg={:.4f}  scost={:.4f}  lossg={:.4f}  " +
                     "p[1]={:.2f}  |g|={:.4f} {:.4f}\t[{:.2f}m / {:.2f}m]\n").format(
                    epoch + (i + 1.0) / N,
                    train_cost / N,
                    train_sparsity_cost / N,
                    train_loss / N,
                    p1 / N,
                    float(gl2_e),
                    float(gl2_g),
                    (time.time() - start_time) / 60.0,
                    (time.time() - start_time) / 60.0 / (i + 1) * N
                ))
                say("\t" + str(["{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                for x in self.encoder.params]) + "\n")
                say("\t" + str(["{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                for x in self.generator.params]) + "\n")

                if dev:
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        # if args.dump and rationale_data:
                        #     self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
                        #                          get_loss_and_pred, sample_generator)
                        #
                        # if args.save_model:
                        #     self.save_model(args.save_model, args)

                    say(("\tsampling devg={:.4f}  mseg={:.4f}" +
                         "  p[1]g={:.2f}  best_dev={:.4f}\n").format(
                        dev_obj,
                        dev_loss,
                        dev_p1,
                        best_dev
                    ))

                    # if rationale_data is not None:
                    #     self.dropout.set_value(0.0)
                    #     r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
                    #         rationale_data, valid_batches_x,
                    #         valid_batches_y, eval_generator)
                    #     self.dropout.set_value(dropout_prob)
                    #     say(("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
                    #          "  prec2={:.4f}\n").format(
                    #         r_mse,
                    #         r_p1,
                    #         r_prec1,
                    #         r_prec2
                    #     ))

            read_output.close()

        metric_output.close()

    def evaluate_data(self, batches_x, batches_y, batches_bv, eval_func, sampling=False):
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        tot_obj, tot_mse, p1 = 0.0, 0.0, 0.0
        dev_v = []
        dev_x = []
        dev_y = []

        for bx, by, bv in zip(batches_x, batches_y, batches_bv):
            if not sampling:
                e, d = eval_func(bx, by)
            else:
                mask = bx != padding_id
                bz, o, e = eval_func(bx, by, bv)
                p1 += np.sum(bz * mask) / (np.sum(mask) + 1e-8)
                tot_obj += o

            tot_mse += e

            dev_v.append(bz)
            dev_y.append(by)
            dev_x.append(bx)

        n = len(batches_x)

        if not sampling:
            return tot_mse / n
        return tot_obj / n, tot_mse / n, p1 / n, dev_v, dev_x, dev_y

    def evaluate_rationale(self, reviews, batches_x, batches_y, eval_func):
        args = self.args
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        aspect = str(args.aspect)
        p1, tot_mse, tot_prec1, tot_prec2 = 0.0, 0.0, 0.0, 0.0
        tot_z, tot_n = 1e-10, 1e-10
        cnt = 0
        for bx, by in zip(batches_x, batches_y):
            mask = bx != padding_id
            bz, o, e, d = eval_func(bx, by)
            tot_mse += e
            p1 += np.sum(bz * mask) / (np.sum(mask) + 1e-8)
            if args.aspect >= 0:
                for z, m in zip(bz.T, mask.T):
                    z = [vz for vz, vm in zip(z, m) if vm]
                    assert len(z) == len(reviews[cnt]["xids"])
                    truez_intvals = reviews[cnt][aspect]
                    prec = sum(1 for i, zi in enumerate(z) if zi > 0 and \
                               any(i >= u[0] and i < u[1] for u in truez_intvals))
                    nz = sum(z)
                    if nz > 0:
                        tot_prec1 += prec / (nz + 0.0)
                        tot_n += 1
                    tot_prec2 += prec
                    tot_z += nz
                    cnt += 1
        # assert cnt == len(reviews)
        n = len(batches_x)
        return tot_mse / n, p1 / n, tot_prec1 / tot_n, tot_prec2 / tot_z

    def dump_rationales(self, path, batches_x, batches_y, eval_func, sample_func):
        embedding_layer = self.embedding_layer
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        lst = []
        for bx, by in zip(batches_x, batches_y):
            loss_vec_r, preds_r, bz = eval_func(bx, by)
            assert len(loss_vec_r) == bx.shape[1]
            for loss_r, p_r, x, y, z in zip(loss_vec_r, preds_r, bx.T, by, bz.T):
                loss_r = float(loss_r)
                p_r, x, y, z = p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
                w = embedding_layer.map_to_words(x)
                r = [u if v == 1 else "__" for u, v in zip(w, z)]
                diff = max(y) - min(y)
                lst.append((diff, loss_r, r, w, x, y, z, p_r))

        # lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
        with open(path, "w") as fout:
            for diff, loss_r, r, w, x, y, z, p_r in lst:
                fout.write(json.dumps({"diff": diff,
                                       "loss_r": loss_r,
                                       "rationale": " ".join(r),
                                       "text": " ".join(w),
                                       "x": x,
                                       "z": z,
                                       "y": y,
                                       "p_r": p_r}) + "\n")


def main():
    assert args.embedding, "Pre-trained word embeddings required."

    vocab = myio.get_vocab(args)
    embedding_layer = myio.create_embedding_layer(args, args.embedding, vocab)

    entities = myio.load_e(args)
    n_classes = len(entities)

    if args.train:
        train_x, train_y, train_e_idxs, train_e = myio.read_docs(args, 'train')

    if args.dev:
        dev_x, dev_y, dev_e_idxs, dev_e = myio.read_docs(args, 'dev')

    if args.train:
        model = Model(
            args=args,
            embedding_layer=embedding_layer,
            nclasses=n_classes
        )
        model.ready()

        model.train(
            (train_x, train_y, train_e_idxs, train_e),
            (dev_x, dev_y, dev_e_idxs, dev_e) if args.dev else None,
            None,  # (test_x, test_y),
            None,
        )


if __name__ == "__main__":
    print theano.config.exception_verbosity
    args = summarization_args.get_args()
    main()
