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
                        default='cnn_w2v_tmp.bin',
                        help='w2v model name and path')

    parser.add_argument('--embedding_file',
                        type=str,
                        default='cnn_w2v_tmp.txt',
                        help='w2v model name and path')

    parser.add_argument('--small_limit',
                        type=int,
                        default=1024,
                        help='small batch limit in number of stories')

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

    parser.add_argument('--embedding_dim',
                        type=int,
                        default=200,
                        help='Size of word vectors')

    parser.add_argument('--train',
                        type=str,
                        default="training_x.json",
                        help='Training Data')

    return parser.parse_args()