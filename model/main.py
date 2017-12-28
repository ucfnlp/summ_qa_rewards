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
from nn.basic import LSTM, apply_dropout
from nn.extended_layers import ExtRCNN, ZLayer, ExtLSTM
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

        dropout = self.dropout = theano.shared(np.float64(args.dropout).astype(theano.config.floatX))

        # len*batch
        x = self.x = T.imatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = []
        layer_type = args.layer.lower()
        for i in xrange(2):
            if layer_type == "rcnn":
                l = RCNN(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation,
                    order=args.order
                )
            elif layer_type == "lstm":
                l = LSTM(
                    n_in=n_e,
                    n_out=n_d,
                    activation=activation
                )
            layers.append(l)

        # len * batch
        self.masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)
        self.word_embs = embs

        flipped_embs = embs[::-1]

        # len*batch*n_d
        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)

        h_final = T.concatenate([h1, h2[::-1]], axis=2)
        h_final = apply_dropout(h_final, dropout)
        size = n_d * 2

        h1_sent = h1[args.sentence_length - 1::args.sentence_length]
        h2_sent = h2[args.sentence_length - 1::args.sentence_length]
        # h_final_sent = T.concatenate([h1_sent, h2_sent[::-1]], axis=2)
        # h_final_sent = apply_dropout(h_final_sent, dropout)

        output_layer = self.output_layer = ZLayer(
            n_in=size,
            n_hidden=args.hidden_dimension2,
            activation=activation
        )

        # sample z given text (i.e. x)
        z_pred, sample_updates = output_layer.sample_all(h_final)

        # we are computing approximated gradient by sampling z;
        # so should mark sampled z not part of the gradient propagation path
        #
        z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
        self.sample_updates = sample_updates
        print "z_pred", z_pred.ndim

        probs_word = output_layer.forward_all(h_final, z_pred)

        # SENTENCE LEVEL

        # output_layer_sent = self.output_layer_sent = ZLayer(
        #     n_in=size,
        #     n_hidden=args.hidden_dimension2,
        #     activation=activation
        # )
        #
        # z_pred_sent, sample_updates_sent = output_layer_sent.sample_all(h_final_sent)
        #
        # z_pred_sent = self.z_pred_sent = theano.gradient.disconnected_grad(z_pred_sent)
        # self.sample_updates_sent = sample_updates_sent
        #
        # probs_sent = output_layer_sent.forward_all(h_final_sent, z_pred_sent)
        #
        # z_pred_sent = T.repeat(z_pred_sent, args.sentence_length, axis=0)
        self.z_pred_combined = z_pred

        # probs_sent = T.repeat(probs_sent, args.sentence_length, axis=0)
        probs = probs_word

        logpz = - T.nnet.binary_crossentropy(probs, self.z_pred_combined) * self.masks
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape)

        # batch
        z = self.z_pred_combined
        self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z[1:] - z[:-1]), axis=0, dtype=theano.config.floatX)

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


class Encoder(object):
    def __init__(self, args, embedding_layer, embedding_layer_y, nclasses, generator):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embedding_layer_y = embedding_layer_y
        self.nclasses = nclasses
        self.generator = generator

    def ready(self):
        generator = self.generator
        embedding_layer = self.embedding_layer
        embedding_layer_y = self.embedding_layer_y

        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = generator.dropout

        # len*batch
        y = self.y = T.imatrix()
        y_mask = T.cast(T.neq(y, padding_id), theano.config.floatX)

        bv = self.bv = T.imatrix()

        z = generator.z_pred_combined
        z = z.dimshuffle((0, 1, "x"))
        y_mask = y_mask.dimshuffle((0, 1, "x"))

        # batch*nclasses
        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        # (len*batch)*n_e
        embs = generator.word_embs
        # (gs_len*batch)*n_e
        embs_y = embedding_layer_y.forward(y.ravel())
        embs_y = embs_y.reshape((y.shape[0], y.shape[1], n_e))

        l = ExtRCNN(
            n_in=n_e,
            n_out=n_d,
            activation=activation,
            order=args.order
        )

        h_prev = embs
        h_prev_y = embs_y
        # len*batch*n_d
        h_next_y = l.forward_all_2(h_prev_y, y_mask)
        h_next_y = theano.gradient.disconnected_grad(h_next_y)

        h_next = l.forward_all(h_prev, z)

        h_next = h_next[::args.sentence_length]
        h_final_y = h_next_y[::args.sentence_length_hl]

        h_final = apply_dropout(h_next, dropout)

        h_final_y = h_final_y.dimshuffle(1, 0, 2) # 15 x 4 x 200
        h_final = h_final.dimshuffle(1, 0, 2) # 15 x 10 x 200

        h_final_y_r = (h_final_y ** 2).sum(2, keepdims=True) # 15 x 4 x 1
        h_final_r = (h_final ** 2).sum(2, keepdims=True).dimshuffle(0,2,1) # 15 x 1 x 10

        batched_dot = T.batched_dot(h_final_y, h_final.dimshuffle(0, 2, 1)) # 15 x 4 x 10

        squared_euclidean_distances = h_final_y_r + h_final_r - 2 * batched_dot # (15 x 4 x 1 + 15 x 1 x 10) +  (15 x 4 x 10)
        similarity = T.sqrt(squared_euclidean_distances).dimshuffle(1,0,2) # 4 x 15 x 10

        loss_mat = self.loss_mat = T.min(similarity, axis=2, keepdims=True) # 4 x 15 x 1

        self.loss_vec = loss_vec = T.mean(loss_mat, axis=0)

        zsum = generator.zsum
        zdiff = generator.zdiff
        logpz = generator.logpz

        padded = T.shape_padaxis(T.zeros_like(bv[0]), axis=1).dimshuffle((1,0))
        component_2 = T.concatenate(
            [bv[1:], padded], axis=0)

        # component_2 = T.stack([shifted_bv, bv], axis=2)
        self.bigram_overlap = component_2 * bv

        intersection = T.sum(self.bigram_overlap)
        jac = (intersection + args.jaccard_smoothing) / (T.sum(bv) + args.jaccard_smoothing)
        jac = 1 - jac

        coherent_factor = args.sparsity * args.coherent
        loss = self.loss = T.mean(loss_vec)
        self.sparsity_cost = T.mean(zsum) * args.sparsity + \
                             T.mean(zdiff) * coherent_factor

        samp = zsum * args.sparsity + zdiff * coherent_factor
        cost_vec = samp + loss_vec + jac
        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))

        self.obj = T.mean(cost_vec) + jac
        self.encoder_params = l.params

        params = self.params = []

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


class Model(object):
    def __init__(self, args, embedding_layer, embedding_layer_y, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embedding_layer_y = embedding_layer_y
        self.nclasses = nclasses

    def ready(self):
        args, embedding_layer_y, embedding_layer, nclasses = self.args, self.embedding_layer_y, self.embedding_layer, self.nclasses
        self.generator = Generator(args, embedding_layer, nclasses)
        self.encoder = Encoder(args, embedding_layer, embedding_layer_y, nclasses, self.generator)
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

        if dev is not None:
            dev_batches_x, dev_batches_y, dev_batches_bv = myio.create_batches(
                dev[0], dev[1], args.batch, padding_id
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
            outputs=[self.encoder.obj, self.encoder.loss, \
                     self.encoder.sparsity_cost, self.z, gnorm_e, gnorm_g],
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

        for epoch in xrange(args.max_epochs):
            read_output = open(args.train_output_readable + '_e_' + str(epoch) + '_sparcity_' + str(args.sparsity) + '.out', 'w+')
            total_words_per_epoch = 0
            total_summaries_per_epoch = 0
            unchanged += 1
            if unchanged > 20:
                metric_output.write("PROBLEM TRAINING, NO DEV IMPROVEMENT")
                metric_output.close()
                return

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

    embedding_layer = myio.create_embedding_layer(args.embedding)
    embedding_layer_y = myio.create_embedding_layer(args.embedding)

    max_len_x = args.sentence_length * args.max_sentences
    max_len_y = args.sentence_length_hl * args.max_sentences_hl

    if args.train:
        train_x, train_y = myio.read_docs(args.train)
        train_x = [embedding_layer.map_to_ids(x)[:max_len_x] for x in train_x]
        train_y = [embedding_layer_y.map_to_ids(y)[:max_len_y] for y in train_y]

    if args.dev:
        dev_x, dev_y = myio.read_docs(args.dev)
        dev_x = [embedding_layer.map_to_ids(x)[:max_len_x] for x in dev_x]
        dev_y = [embedding_layer_y.map_to_ids(y)[:max_len_y] for y in dev_y]

    if args.load_rationale:
        rationale_data = myio.read_rationales(args.load_rationale)
        for x in rationale_data:
            x["xids"] = embedding_layer.map_to_ids(x["x"])

    if args.train:
        model = Model(
            args=args,
            embedding_layer=embedding_layer,
            embedding_layer_y=embedding_layer_y,
            nclasses=len(train_y[0])
        )
        model.ready()

        # debug_func2 = theano.function(
        #        inputs = [ model.x, model.z ],
        #        outputs = model.generator.logpz
        #    )
        # theano.printing.debugprint(debug_func2)
        # return

        model.train(
            (train_x, train_y),
            (dev_x, dev_y) if args.dev else None,
            None,  # (test_x, test_y),
            rationale_data if args.load_rationale else None
        )

    if args.load_model and args.dev and not args.train:
        model = Model(
            args=None,
            embedding_layer=embedding_layer,
            nclasses=-1
        )
        model.load_model(args.load_model)
        say("model loaded successfully.\n")

        # compile an evaluation function
        eval_func = theano.function(
            inputs=[model.x, model.y],
            outputs=[model.z, model.encoder.obj, model.encoder.loss,
                     model.encoder.pred_diff],
            updates=model.generator.sample_updates
        )

        # compile a predictor function
        pred_func = theano.function(
            inputs=[model.x],
            outputs=[model.z, model.encoder.preds],
            updates=model.generator.sample_updates
        )

        # batching data
        padding_id = embedding_layer.vocab_map["<padding>"]
        dev_batches_x, dev_batches_y = myio.create_batches(
            dev_x, dev_y, args.batch, padding_id
        )

        # disable dropout
        model.dropout.set_value(0.0)
        dev_obj, dev_loss, dev_diff, dev_p1 = model.evaluate_data(
            dev_batches_x, dev_batches_y, eval_func, sampling=True)
        say("{} {} {} {}\n".format(dev_obj, dev_loss, dev_diff, dev_p1))


if __name__ == "__main__":
    print theano.config.exception_verbosity
    args = summarization_args.get_args()
    main()
