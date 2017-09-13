import argparse
import theano

_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true','True', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--embedding',
                        type=str,
                        default='cnn_w2v_tmp.txt',
                        help='w2v model name and path')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='Dropout Rate')

    parser.add_argument('--activation',
                        type=str,
                        default='tanh',
                        help='Activation')

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

    return parser.parse_args()