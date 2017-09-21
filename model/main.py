
import os, gzip
import time
import json
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn.optimization import create_optimization_updates
from nn.basic import sigmoid, LSTM, apply_dropout
from nn.advanced import Layer, RCNN
from util import say
import myio
import summarization_args
from nn.extended_layers import ExtRCNN, ExtLSTM, ZLayer
from data import process_data


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
        activation = 'tanh'

        layers = self.layers = [ ]
        layer_type = args.layer.lower()
        for i in xrange(2):
            if layer_type == "rcnn":
                l = RCNN(
                        n_in = n_e,# if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order
                    )
            elif layer_type == "lstm":
                l = LSTM(
                        n_in = n_e,# if i == 0 else n_d,
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

        flipped_embs = embs[::-1]

        # len*bacth*n_d
        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)

        h_final = T.concatenate([h1, h2[::-1]], axis=2)
        h_final = apply_dropout(h_final, dropout)
        size = n_d * 2

        h_final_sent = T.concatenate([h1[args.sentence_length - 1::args.sentence_length],
                                      h2[::args.sentence_length]], axis=2)

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


        # SENTENCE LEVEL

        output_layer_sent = self.output_layer_sent = ZLayer(
                n_in = size,
                n_hidden = args.hidden_dimension2,
                activation = activation
            )

        z_pred_sent, sample_updates_sent = output_layer_sent.sample_all(h_final_sent)

        z_pred_sent = self.z_pred_sent = theano.gradient.disconnected_grad(z_pred)
        self.sample_updates_sent = sample_updates_sent

        probs = output_layer.forward_all(h_final, z_pred)

        self.z_pred_combined = z_pred * T.repeat(z_pred_sent, args.sentence_length)

        logpz = - T.nnet.binary_crossentropy(probs, z_pred) * masks
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape)

        # batch
        z = z_pred
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

    def __init__(self, args, embedding_layer, nclasses, generator):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.generator = generator

    def ready(self):
        generator = self.generator
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # len*batch
        x = self.x = T.imatrix()

        z = generator.z_pred_combined

        z = z.dimshuffle((0,1,"x"))

        # max_gs_len * batch
        y = self.y = T.fmatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = 'tanh'

        layers = self.layers = [ ]
        depth = args.depth
        layer_type = args.layer.lower()
        for i in xrange(depth):
            if layer_type == "rcnn":
                l = ExtRCNN(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order
                    )
            elif layer_type == "lstm":
                l = ExtLSTM(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            layers.append(l)

        z_y = T.ones_like(y.dimshuffle((0, 1, "x")))

        # len * batch * 1
        masks = T.cast(T.neq(x, padding_id).dimshuffle((0,1,"x")) * z, theano.config.floatX)
        # gs_len * batch * 1
        masks_y = T.cast(T.neq(y, padding_id).dimshuffle((0, 1, "x")) * z_y, theano.config.floatX)

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # (gs_len*batch)*n_e
        embs_y = embedding_layer.forward(y.ravel()).reshape((y.shape[0], y.shape[1], n_e))

        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)

        lst_states = [ ]
        lst_states_y = []

        h_prev = embs
        h_prev_y = embs_y

        for l in layers:
            # len*batch*n_d
            h_next = l.forward_all(h_prev, z, return_c=True)
            # len_gs*batch*n_d
            h_next_y = l.forward_all(h_prev_y, z_y, return_c=True)

            lst_states.append(h_next[::args.sentence_length])
            lst_states_y.append(h_next_y[::args.sentence_length])

            h_prev = apply_dropout(h_next, dropout)
            h_prev_y = h_next_y


        h_final = lst_states[-1]
        h_final_y = lst_states_y[-1]

        h_final = apply_dropout(h_final, dropout)


        loss = self.loss = T.sum()



        params = self.params = [ ]
        for l in layers:
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

        cost = self.cost = loss * 10 + l2_cost


class Model(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
        self.generator = Generator(args, embedding_layer, nclasses)
        self.encoder = Encoder(args, embedding_layer, nclasses, self.generator)
        self.generator.ready()
        self.encoder.ready()
        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.y = self.encoder.y
        self.z = self.generator.z_pred_combined
        self.params = self.encoder.params + self.generator.params

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
                ([ x.get_value() for x in self.encoder.params ],   # encoder
                 [ x.get_value() for x in self.generator.params ], # generator
                 self.nclasses,
                 args                                 # training configuration
                ),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            eparams, gparams, nclasses, args  = pickle.load(fin)

        # construct model/network using saved configuration
        self.args = args
        self.nclasses = nclasses
        self.ready()
        for x,v in zip(self.encoder.params, eparams):
            x.set_value(v)
        for x,v in zip(self.generator.params, gparams):
            x.set_value(v)

    def train(self, train, dev, test, rationale_data):
        args = self.args
        dropout = self.dropout
        padding_id = self.embedding_layer.vocab_map["<padding>"]

        if dev is not None:
            dev_batches_x, dev_batches_y = myio.create_batches(
                            dev[0], dev[1], args.batch, padding_id
                        )
        if test is not None:
            test_batches_x, test_batches_y = myio.create_batches(
                            test[0], test[1], args.batch, padding_id
                        )
        if rationale_data is not None:
            valid_batches_x, valid_batches_y = myio.create_batches(
                    [ u["xids"] for u in rationale_data ],
                    [ u["y"] for u in rationale_data ],
                    args.batch,
                    padding_id,
                    sort = False
                )

        start_time = time.time()
        train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )
        say("{:.2f}s to create training batches\n\n".format(
                time.time()-start_time
            ))

        updates_e, lr_e, gnorm_e = create_optimization_updates(
                               cost = self.generator.cost_e,
                               params = self.encoder.params,
                               method = args.learning,
                               lr = args.learning_rate
                        )[:3]


        updates_g, lr_g, gnorm_g = create_optimization_updates(
                               cost = self.generator.cost,
                               params = self.generator.params,
                               method = args.learning,
                               lr = args.learning_rate
                        )[:3]

        sample_generator = theano.function(
                inputs = [ self.x ],
                outputs = self.z_pred,
                #updates = self.generator.sample_updates
                #allow_input_downcast = True
            )

        get_loss_and_pred = theano.function(
                inputs = [ self.x, self.z, self.y ],
                outputs = [ self.generator.loss_vec, self.encoder.preds ]
            )

        eval_generator = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.z, self.generator.obj, self.generator.loss,
                                self.encoder.pred_diff ],
                givens = {
                    self.z : self.generator.z_pred_combined
                },
                #updates = self.generator.sample_updates,
                #no_default_updates = True
            )

        train_generator = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.generator.obj, self.generator.loss, \
                                self.generator.sparsity_cost, self.z, gnorm_g, gnorm_e ],
                givens = {
                    self.z : self.generator.z_pred_combined
                },
                #updates = updates_g,
                updates = updates_g.items() + updates_e.items() #+ self.generator.sample_updates,
                #no_default_updates = True
            )

        eval_period = args.eval_period
        unchanged = 0
        best_dev = 1e+2
        best_dev_e = 1e+2
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 10: return

            train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )

            processed = 0
            train_cost = 0.0
            train_loss = 0.0
            train_sparsity_cost = 0.0
            p1 = 0.0
            start_time = time.time()

            N = len(train_batches_x)
            for i in xrange(N):
                if (i+1) % 100 == 0:
                    say("\r{}/{}     ".format(i+1,N))

                bx, by = train_batches_x[i], train_batches_y[i]
                mask = bx != padding_id

                cost, loss, sparsity_cost, bz, gl2_g, gl2_e = train_generator(bx, by)

                k = len(by)
                processed += k
                train_cost += cost
                train_loss += loss
                train_sparsity_cost += sparsity_cost
                p1 += np.sum(bz*mask) / (np.sum(mask)+1e-8)

                if (i == N-1) or (eval_period > 0 and processed/eval_period >
                                    (processed-k)/eval_period):
                    say("\n")
                    say(("Generator Epoch {:.2f}  costg={:.4f}  scost={:.4f}  lossg={:.4f}  " +
                        "p[1]={:.2f}  |g|={:.4f} {:.4f}\t[{:.2f}m / {:.2f}m]\n").format(
                            epoch+(i+1.0)/N,
                            train_cost / (i+1),
                            train_sparsity_cost / (i+1),
                            train_loss / (i+1),
                            p1 / (i+1),
                            float(gl2_g),
                            float(gl2_e),
                            (time.time()-start_time)/60.0,
                            (time.time()-start_time)/60.0/(i+1)*N
                        ))
                    say("\t"+str([ "{:.1f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                    for x in self.encoder.params ])+"\n")
                    say("\t"+str([ "{:.1f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                    for x in self.generator.params ])+"\n")

                    if dev:
                        self.dropout.set_value(0.0)
                        dev_obj, dev_loss, dev_diff, dev_p1 = self.evaluate_data(
                                dev_batches_x, dev_batches_y, eval_generator, sampling=True)

                        if dev_obj < best_dev:
                            best_dev = dev_obj
                            unchanged = 0
                            if args.dump and rationale_data:
                                self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
                                            get_loss_and_pred, sample_generator)

                            if args.save_model:
                                self.save_model(args.save_model, args)

                        say(("\tsampling devg={:.4f}  mseg={:.4f}  avg_diffg={:.4f}" +
                                    "  p[1]g={:.2f}  best_dev={:.4f}\n").format(
                            dev_obj,
                            dev_loss,
                            dev_diff,
                            dev_p1,
                            best_dev
                        ))

                        if rationale_data is not None:
                            r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
                                    rationale_data, valid_batches_x,
                                    valid_batches_y, eval_generator)
                            say(("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
                                        "  prec2={:.4f}\n").format(
                                    r_mse,
                                    r_p1,
                                    r_prec1,
                                    r_prec2
                            ))

                        self.dropout.set_value(dropout_prob)



    def evaluate_data(self, batches_x, batches_y, eval_func, sampling=False):
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        tot_obj, tot_mse, tot_diff, p1 = 0.0, 0.0, 0.0, 0.0
        for bx, by in zip(batches_x, batches_y):
            if not sampling:
                e, d = eval_func(bx, by)
            else:
                mask = bx != padding_id
                bz, o, e, d = eval_func(bx, by)
                p1 += np.sum(bz*mask) / (np.sum(mask) + 1e-8)
                tot_obj += o
            tot_mse += e
            tot_diff += d
        n = len(batches_x)
        if not sampling:
            return tot_mse/n, tot_diff/n
        return tot_obj/n, tot_mse/n, tot_diff/n, p1/n

    def evaluate_rationale(self, reviews, batches_x, batches_y, eval_func):
        args = self.args
        assert args.aspect >= 0
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        aspect = str(args.aspect)
        p1, tot_mse, tot_prec1, tot_prec2 = 0.0, 0.0, 0.0, 0.0
        tot_z, tot_n = 1e-10, 1e-10
        cnt = 0
        for bx, by in zip(batches_x, batches_y):
            mask = bx != padding_id
            bz, o, e, d = eval_func(bx, by)
            tot_mse += e
            p1 += np.sum(bz*mask)/(np.sum(mask) + 1e-8)
            for z,m in zip(bz.T, mask.T):
                z = [ vz for vz,vm in zip(z,m) if vm ]
                assert len(z) == len(reviews[cnt]["xids"])
                truez_intvals = reviews[cnt][aspect]
                prec = sum( 1 for i, zi in enumerate(z) if zi>0 and \
                            any(i>=u[0] and i<u[1] for u in truez_intvals) )
                nz = sum(z)
                if nz > 0:
                    tot_prec1 += prec/(nz+0.0)
                    tot_n += 1
                tot_prec2 += prec
                tot_z += nz
                cnt += 1
        assert cnt == len(reviews)
        n = len(batches_x)
        return tot_mse/n, p1/n, tot_prec1/tot_n, tot_prec2/tot_z

    def dump_rationales(self, path, batches_x, batches_y, eval_func, sample_func):
        embedding_layer = self.embedding_layer
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        lst = [ ]
        for bx, by in zip(batches_x, batches_y):
            bz = np.ones(bx.shape, dtype="int8")
            loss_vec_t, preds_t = eval_func(bx, bz, by)
            bz = sample_func(bx)
            loss_vec_r, preds_r = eval_func(bx, bz, by)
            assert len(loss_vec_r) == bx.shape[1]
            for loss_t, p_t, loss_r, p_r, x,y,z in zip(loss_vec_t, preds_t, \
                                loss_vec_r, preds_r, bx.T, by, bz.T):
                loss_t, loss_r = float(loss_t), float(loss_r)
                p_t, p_r, x, y, z = p_t.tolist(), p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
                w = embedding_layer.map_to_words(x)
                r = [ u if v == 1 else "__" for u,v in zip(w,z) ]
                diff = max(y)-min(y)
                lst.append((diff, loss_t, loss_r, r, w, x, y, z, p_t, p_r))

        #lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
        with open(path,"w") as fout:
            for diff, loss_t, loss_r, r, w, x, y, z, p_t, p_r in lst:
                fout.write( json.dumps( { "diff": diff,
                                          "loss_t": loss_t,
                                          "loss_r": loss_r,
                                          "rationale": " ".join(r),
                                          "text": " ".join(w),
                                          "x": x,
                                          "z": z,
                                          "y": y,
                                          "p_t": p_t,
                                          "p_r": p_r } ) + "\n" )


def main():
    print args
    assert args.embedding, "Pre-trained word embeddings required."

    embedding_layer = myio.create_embedding_layer(args.embedding)

    max_len = args.sentence_length*args.max_sentences

    if args.train:
        train_x, train_y = myio.read_docs(args)
        train_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in train_x ]

    if args.dev:
        dev_x, dev_y = myio.read_annotations(args.dev)
        dev_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in dev_x ]

    if args.load_rationale:
        rationale_data = myio.read_rationales(args.load_rationale)
        for x in rationale_data:
            x["xids"] = embedding_layer.map_to_ids(x["x"])

    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = len(train_y[0])
                )
        model.ready()

        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                None, #(test_x, test_y),
                rationale_data if args.load_rationale else None
            )

    if args.load_model and args.dev and not args.train:
        model = Model(
                    args = None,
                    embedding_layer = embedding_layer,
                    nclasses = -1
                )
        model.load_model(args.load_model)
        say("model loaded successfully.\n")

        # compile an evaluation function
        eval_func = theano.function(
                inputs = [ model.x, model.y ],
                outputs = [ model.z, model.generator.obj, model.generator.loss,
                                model.encoder.pred_diff ],
                givens = {
                    model.z : model.generator.z_pred
                },
            )

        # compile a predictor function
        pred_func = theano.function(
                inputs = [ model.x ],
                outputs = [ model.z, model.encoder.preds ],
                givens = {
                    model.z : model.generator.z_pred
                },
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


if __name__=="__main__":
    args = summarization_args.get_args()
    main()