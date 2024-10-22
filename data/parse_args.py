import argparse
import theano

_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true','True', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # URLS FOR SPLITS

    parser.add_argument('--train_urls',
                        type=str,
                        default="lists/all_train.txt",
                        help='Training URLs')

    parser.add_argument('--test_urls',
                        type=str,
                        default="lists/all_test.txt",
                        help='Test set URLs')

    parser.add_argument('--dev_urls',
                        type=str,
                        default="lists/all_val.txt",
                        help='Dev Set URLs')

    parser.add_argument('--raw_data',
                        type=str,
                        default='cnn/stories/',
                        help='Raw data CNN')

    parser.add_argument('--parsed_output_loc',
                        type=str,
                        default='parse/cnn/',
                        help='File path for parsed data')

    parser.add_argument('--stopwords',
                        type=str,
                        default='stopwords.txt',
                        help='List of stopwords')

    parser.add_argument('--source',
                        type=str,
                        default='cnn',
                        help='Data source')

    return parser.parse_args()