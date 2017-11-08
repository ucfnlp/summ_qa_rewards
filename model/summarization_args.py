import argparse

import theano

_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true', 'True', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--embedding',
                        type=str,
                        default='../data/cnn_w2v_tmp.txt',
                        help='w2v model name and path')

    parser.add_argument('--embedding_dim',
                        type=int,
                        default=200,
                        help='Size of word vectors')

    parser.add_argument('--sentence_length',
                        type=int,
                        default=30,
                        help='length of single sentence')

    parser.add_argument('--max_sentences',
                        type=int,
                        default=10,
                        help='Size of document in sentences')

    parser.add_argument('--sentence_length_hl',
                        type=int,
                        default=15,
                        help='length of single sentence in highlights')

    parser.add_argument('--max_sentences_hl',
                        type=int,
                        default=4,
                        help='Number of total sentences for highlights')

    parser.add_argument("--train_output_readable",
                        type=str,
                        default="../data/results/results",
                        help="path to annotated train results"
                        )
    parser.add_argument("--load_rationale",
                        type=str,
                        default="",
                        help="path to annotated rationale data"
                        )

    parser.add_argument("--save_model",
                        type=str,
                        default="",
                        help="path to save model parameters"
                        )
    parser.add_argument("--load_model",
                        type=str,
                        default="",
                        help="path to load model"
                        )
    parser.add_argument("--train",
                        type=str,
                        default="../data/training_x.json",
                        help="path to training data"
                        )
    parser.add_argument("--dev",
                        type=str,
                        default="",
                        help="path to development data"
                        )
    parser.add_argument("--test",
                        type=str,
                        default="",
                        help="path to test data"
                        )
    parser.add_argument("--dump",
                        type=str,
                        default="",
                        help="path to dump rationale"
                        )
    parser.add_argument("--max_epochs",
                        type=int,
                        default=30,
                        help="maximum # of epochs"
                        )
    parser.add_argument("--eval_period",
                        type=int,
                        default=-1,
                        help="evaluate model every k examples"
                        )
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="mini-batch size"
                        )
    parser.add_argument("--learning",
                        type=str,
                        default="adam",
                        help="learning method"
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.0005,
                        help="learning rate"
                        )
    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="dropout probability"
                        )
    parser.add_argument("--l2_reg",
                        type=float,
                        default=1e-6,
                        help="L2 regularization weight"
                        )
    parser.add_argument("-act", "--activation",
                        type=str,
                        default="tanh",
                        help="type of activatioin function"
                        )
    parser.add_argument("-d", "--hidden_dimension",
                        type=int,
                        default=200,
                        help="hidden dimension"
                        )
    parser.add_argument("-d2", "--hidden_dimension2",
                        type=int,
                        default=30,
                        help="hidden dimension"
                        )

    parser.add_argument("--layer",
                        type=str,
                        default="rcnn",
                        help="type of recurrent layer"
                        )
    parser.add_argument("--depth",
                        type=int,
                        default=2,
                        help="number of layers"
                        )
    parser.add_argument("--pooling",
                        type=int,
                        default=0,
                        help="whether to use mean pooling or the last state"
                        )
    parser.add_argument("--order",
                        type=int,
                        default=2,
                        help="feature filter width"
                        )
    parser.add_argument("--use_all",
                        type=int,
                        default=1,
                        help="whether to use the states of all layers"
                        )

    parser.add_argument("--sparsity",
                        type=float,
                        default=0.0003
                        )
    parser.add_argument("--coherent",
                        type=float,
                        default=2.0
                        )
    parser.add_argument("--aspect",
                        type=int,
                        default=-1
                        )
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9
                        )
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999
                        )
    parser.add_argument("--decay_lr",
                        type=int,
                        default=1
                        )
    parser.add_argument("--fix_emb",
                        type=int,
                        default=1
                        )

    return parser.parse_args()
