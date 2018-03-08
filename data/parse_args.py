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

    parser.add_argument('--raw_data_cnn',
                        type=str,
                        default='cnn/stories/',
                        help='Raw data CNN')

    parser.add_argument('--parsed_output_loc',
                        type=str,
                        default='parse/cnn/',
                        help='File path for parsed data')

    parser.add_argument('--cp',
                        type=str,
                        default='',
                        help='CLASSPATH')

    parser.add_argument('--sp',
                        type=str,
                        default='',
                        help='STANFORD_PARSER')

    parser.add_argument('--sm',
                        type=str,
                        default='',
                        help='STANFORD_MODEL')

    parser.add_argument('--out_limit',
                        type=int,
                        default=5000,
                        help='Max sentences per file')

    parser.add_argument('--process',
                        type='bool',
                        default=False,
                        help='Process raw input for SCNLP')

    return parser.parse_args()