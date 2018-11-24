import cPickle as pickle
import gzip
import json
import os
import time
import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import sklearn.metrics as sk
import theano

import myio
import summarization_args

from nn.optimization import create_optimization_updates
from nn.encoder import QAEncoder as Encoder
from util import say


class Model(object):
    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer,  self.nclasses

        self.encoder = Encoder(args, nclasses, embedding_layer)
        self.encoder.ready()

        self.dropout = self.encoder.dropout
        if not args.qa_hl_only:
            self.x = self.encoder.x
        self.y = self.encoder.y
        self.gold_standard_entities = self.encoder.gold_standard_entities

        self.params = self.encoder.params

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
            loaded = pickle.load(fin)

            eparams = loaded[0]
            args = loaded[1]

            self.args = args

        self.ready()
        for x, v in zip(self.encoder.params, eparams):
            x.set_value(v)

    def dev_full(self):
        inputs_d = [self.x, self.generator.posit_x, self.y, self.bm, self.gold_standard_entities, self.fw_mask,
                    self.chunk_sizes, self.encoder.loss_mask]

        eval_generator = theano.function(
            inputs=inputs_d,
            outputs=[self.generator.non_sampled_zpred, self.encoder.obj, self.encoder.loss, self.encoder.preds_clipped],
            on_unused_input='ignore'
        )

        self.dropout.set_value(0.0)

        dev_obj, dev_z, dev_x, dev_sha, dev_acc, _ = self.evaluate_data(eval_generator)

        myio.save_dev_results(self.args, None, dev_z, dev_x, dev_sha)
        myio.get_rouge(self.args)

    def test(self):
        outputs = [self.encoder.loss, self.encoder.preds_clipped]

        if args.qa_entity_output:
            filename = myio.create_json_filename_qa(args).replace('.json', '_test.json')
            ofp_test = open(filename, 'w+')

        json_test = dict()

        json_test['sha'] = []
        json_test['pred'] = []
        json_test['loss_mask'] = []
        json_test['ground_truth'] = []

        if args.qa_hl_only:
            inputs = [self.y, self.gold_standard_entities, self.encoder.loss_mask]
        else:
            inputs = [self.x, self.y, self.gold_standard_entities, self.encoder.loss_mask]

        eval_model = theano.function(
            inputs=inputs,
            outputs=outputs,
            on_unused_input='ignore'
        )

        self.dropout.set_value(0.0)

        tot_obj = 0.0
        N = 0

        acc = []
        f1 = []

        num_files = self.args.num_files_test

        for i in xrange(num_files):

            batches_x, batches_y, batches_e, batches_sha = myio.load_batches(
                self.args.batch_dir + self.args.source + 'test', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                bx, by, be = batches_x[j], batches_y[j], batches_e[j]

                be, ble = myio.create_1h(be, args.nclasses, args.n, args.pad_repeat)
                if args.qa_hl_only:
                    o, preds = eval_model(by, be, ble)
                else:
                    o, preds = eval_model(bx, by, be, ble)

                if args.qa_entity_output:
                    json_test['sha'].extend([item for ls in batches_sha[j] for item in ls])
                    json_test['pred'].extend([item for item in np.ndarray.tolist(preds)])

                tot_obj += o

                acc_, f1_ = self.eval_qa(be, preds, ble)
                acc.append(acc_)
                f1.append(f1_)

            N += cur_len

        print 'TEST set results:'
        print ' Accuracy :', np.mean(acc)

        if args.qa_entity_output:
            json.dump(json_test, ofp_test)
            ofp_test.close()

    def train(self):
        args = self.args

        updates_e, lr_e, gnorm_e = create_optimization_updates(
            cost=self.encoder.cost_e,
            params=self.encoder.params,
            method=args.learning,
            beta1=args.beta1,
            beta2=args.beta2,
            lr=args.learning_rate
        )[:3]

        outputs_d = [self.encoder.loss, self.encoder.preds_clipped]
        outputs_t = [self.encoder.loss, self.encoder.loss_vec, self.encoder.preds_clipped]

        if args.qa_hl_only:
            inputs_d = [self.y, self.gold_standard_entities, self.encoder.loss_mask]
            inputs_t = [self.y, self.gold_standard_entities, self.encoder.loss_mask]
        else:
            inputs_d = [self.x, self.y, self.gold_standard_entities, self.encoder.loss_mask]
            inputs_t = [self.x, self.y, self.gold_standard_entities, self.encoder.loss_mask]

        eval_generator = theano.function(
            inputs=inputs_d,
            outputs=outputs_d,
            on_unused_input='ignore'
        )

        train_generator = theano.function(
            inputs=inputs_t,
            outputs=outputs_t,
            updates=updates_e.items(),
            on_unused_input='ignore'
        )

        say("Model Built Full (QA)\n\n")

        unchanged = 0
        best_dev = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.10 + 1e-3
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        filename = myio.create_json_filename_qa(args)
        ofp_train = open(filename, 'w+')

        json_train = dict()
        best_train_output = None

        random.seed(datetime.now())

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
                train_loss = 0.0
                more_count += 1

                if more_count > 5:
                    break
                start_time = time.time()

                loss_all = []
                loss_vec_all = []
                train_acc = []
                train_f1 = []

                num_files = args.num_files_train
                N = args.online_batch_size * num_files

                cur_train_output = dict()
                cur_train_output['sha'] = []
                cur_train_output['pred'] = []

                for i in xrange(num_files):

                    train_batches_x, train_batches_y, train_batches_e, train_batches_sha = myio.load_batches(
                        args.batch_dir + args.source + 'train', i)

                    cur_len = len(train_batches_x)

                    perm2 = range(cur_len)
                    random.shuffle(perm2)

                    train_batches_x = [train_batches_x[k] for k in perm2]
                    train_batches_y = [train_batches_y[k] for k in perm2]
                    train_batches_e = [train_batches_e[k] for k in perm2]
                    train_batches_sha = [train_batches_sha[k] for k in perm2]

                    for j in xrange(cur_len):
                        if args.full_test:
                            if (i * args.online_batch_size + j + 1) % 10 == 0:
                                say("\r{}/{}       ".format(i * args.online_batch_size + j + 1, N))

                        bx, by, be = train_batches_x[j], train_batches_y[j], train_batches_e[j]

                        be, blm = myio.create_1h(be, args.nclasses, args.n, args.pad_repeat)
                        # self.x, self.y, self.gold_standard_entities, self.encoder.loss_mask
                        if args.qa_hl_only:
                            loss, loss_vec, preds_tr = train_generator(by, be, blm)
                        else:
                            loss, loss_vec, preds_tr = train_generator(bx, by, be, blm)

                        acc, f1 = self.eval_qa(be, preds_tr, blm)

                        train_acc.append(acc)
                        train_f1.append(f1)
                        loss_all.append(loss)
                        loss_vec_all.append(np.mean(loss_vec))

                        train_loss += loss

                cur_train_avg_cost = train_loss / N

                if args.dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_acc, dev_f1 = self.evaluate_data(eval_generator)
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
                    lr_val = lr_e.get_value() * 0.5
                    lr_val = np.float64(lr_val).astype(theano.config.floatX)

                    lr_e.set_value(lr_val)
                    say("Decrease learning rate to {}\n".format(float(lr_val)))
                    for p, v in zip(self.params, param_bak):
                        p.set_value(v)
                    continue

                myio.record_observations_verbose_qa(json_train, epoch + 1, loss_all, loss_vec_all,
                                                 dev_acc, dev_f1, np.mean(train_acc), np.mean(train_f1))

                last_train_avg_cost = cur_train_avg_cost

                say("\n")
                say(("Generator Epoch {:.2f} lossg={:.4f}  " +
                     "\t[{:.2f}m / {:.2f}m]\n").format(
                    epoch + 1,
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
                            filename = args.save_model + myio.create_fname_identifier_qa(args)
                            self.save_model(filename, args)

                            json_train['BEST_DEV_EPOCH'] = epoch

                            if args.qa_entity_output:
                                best_train_output = cur_train_output

            if more_count > 5:
                json_train['ERROR'] = 'Stuck reducing error rate, at epoch ' + str(epoch + 1) + '. LR = ' + str(lr_val)
                json.dump(json_train, ofp_train)
                ofp_train.close()
                return

        if unchanged > 20:
            json_train['UNCHANGED'] = unchanged

        if args.qa_entity_output:
            json_train['TRAIN_ENTITY_OUTPUT'] = best_train_output

        json.dump(json_train, ofp_train)
        ofp_train.close()

    def evaluate_data(self, eval_func):
        tot_obj = 0.0
        N = 0

        dev_acc = []
        dev_f1 = []

        num_files = self.args.num_files_dev

        for i in xrange(num_files):

            batches_x, batches_y, batches_e, _ = myio.load_batches(
                self.args.batch_dir + self.args.source + 'dev', i)

            cur_len = len(batches_x)

            for j in xrange(cur_len):
                bx, by, be = batches_x[j], batches_y[j], batches_e[j]

                be, ble = myio.create_1h(be, args.nclasses, args.n, args.pad_repeat)
                if args.qa_hl_only:
                    o, preds = eval_func(by, be, ble)
                else:
                    o, preds = eval_func(bx, by, be, ble)

                tot_obj += o

                acc, f1 = self.eval_qa(be, preds, ble)
                dev_acc.append(acc)
                dev_f1.append(f1)

            N += cur_len

        return tot_obj / float(N), np.mean(dev_acc), np.mean(dev_f1)

    def eval_qa(self, gs, preds, valid_mask):
        system = np.argmax(preds, axis=1)
        valid_mask = np.ndarray.flatten(valid_mask)

        valid_gs = []
        valid_sy = []

        for system_pred, gold_standard_val, mask in zip(system, gs, valid_mask):
            if mask > 0:
                valid_gs.append(gold_standard_val)
                valid_sy.append(system_pred)

        accuracy_score = sk.accuracy_score(valid_gs, valid_sy)
        f1_score = sk.f1_score(valid_gs, valid_sy, average='micro')

        return accuracy_score, f1_score


def main():
    assert args.embedding, "Pre-trained word embeddings required."

    vocab = myio.get_vocab(args)
    embedding_layer = myio.create_embedding_layer(args, args.embedding, vocab, args.embedding_dim, '<unk>')

    n_classes = args.nclasses

    model = Model(
        args=args,
        embedding_layer=embedding_layer,
        nclasses=n_classes
    )

    if args.train:
        model.ready()
        model.train()

    elif args.dev:
        model.dev_full()

    elif args.test:
        model.load_model(args.save_model + args.load_model)
        model.test()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = summarization_args.get_args()
    main()