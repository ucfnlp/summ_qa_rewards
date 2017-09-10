import argparse
import theano

_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true','True', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--full_test',
                        type='bool',
                        default=False,
                        help='Process full selection of CNN data')

    parser.add_argument('--raw_data',
                        type=str,
                        default='cnn',
                        help='Raw CNN data')

    parser.add_argument('--word_model',
                        type=str,
                        default='cnn_w2v_tmp',
                        help='w2v model name and path')

    parser.add_argument('--small_limit',
                        type=int,
                        default=30,
                        help='small batch limit')

    parser.add_argument('--embedding_dim',
                        type=int,
                        default=200,
                        help='Size of word vectors')

    return parser.parse_args()