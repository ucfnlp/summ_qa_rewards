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
                        default=True,
                        help='Process full selection of CNN data')

    parser.add_argument('--pipeline',
                        type='bool',
                        default=False,
                        help='Process for Stanford Core NLP')

    parser.add_argument('--raw_data_cnn',
                        type=str,
                        default='/data1/corpora/cnn_dailymail/cnn-dailymail/cnn_stories_tokenized/',
                        # default='cnn/stories/',
                        help='Raw data CNN')

    parser.add_argument('--raw_data_dm',
                        type=str,
                        default='/data1/corpora/cnn_dailymail/cnn-dailymail/dm_stories_tokenized/',
                        # default='cnn/stories/',
                        help='Raw data Daily Mail')

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
                        default=1000,
                        help='small batch limit in number of stories')

    parser.add_argument('--inp_len',
                        type=int,
                        default=400,
                        help='length of single sample')

    parser.add_argument('--inp_len_hl',
                        type=int,
                        default=25,
                        help='length of single sentence in highlights')

    parser.add_argument('--max_sentences_hl',
                        type=int,
                        default=4,
                        help='Number of total sentences for highlights')

    parser.add_argument('--embedding_dim',
                        type=int,
                        default=200,
                        help='Size of word vectors')

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

    parser.add_argument('--train',
                        type=str,
                        default="training.json",
                        help='Training Data')

    parser.add_argument('--dev',
                        type=str,
                        default="dev.json",
                        help='Development Data')

    parser.add_argument('--test',
                        type=str,
                        default="test.json",
                        help='Test Data')

    parser.add_argument('--intermediate',
                        type=str,
                        default="stanford_nlp",
                        help='file location for StanfordCoreNLP input')

    parser.add_argument('--placeholder',
                        type=int,
                        default=2,
                        help='IDX of <placeholder> token')

    parser.add_argument('--unk',
                        type=int,
                        default=1,
                        help='IDX of <unk> token')

    parser.add_argument('--stgz',
                        type=str,
                        default="/Users/kristjan/Documents/Grad School/CAP7919/danqi/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz",
                        help='')

    parser.add_argument('--stjar',
                        type=str,
                        default="/Users/kristjan/Documents/Grad School/CAP7919/danqi/stanford-ner/stanford-ner.jar",
                        help='')

    return parser.parse_args()