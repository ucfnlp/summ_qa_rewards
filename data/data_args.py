import argparse
import theano

_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true','True', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # GENERAL DATA/DEBUG INFO

    parser.add_argument('--full_test',
                        type='bool',
                        default=True,
                        help='Process full selection of CNN data')

    parser.add_argument('--pipeline',
                        type='bool',
                        default=False,
                        help='Process for Stanford Core NLP')

    parser.add_argument('--split_by_source',
                        type='bool',
                        default=False,
                        help='Split data further into domains')

    parser.add_argument('--vocab_size',
                        type=int,
                        default=100000,
                        help='Vocab size')

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

    parser.add_argument('--small_limit',
                        type=int,
                        default=4096,
                        help='small batch limit in number of stories')

    # MODEL INPUT INFO

    parser.add_argument('--word_model',
                        type=str,
                        default='cnn_w2v_tmp.bin',
                        help='w2v model name and path')

    parser.add_argument('--embedding_file',
                        type=str,
                        default='emb/glove.6B.200d.txt',
                        help='w2v model name and path')

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

    parser.add_argument('--n',
                        type=int,
                        default=4,
                        help='Number of highlights to use for model input')

    parser.add_argument('--use_root',
                        type='bool',
                        default=True,
                        help='Whether to use the ROOT based entity in the data.')

    parser.add_argument('--use_person',
                        type='bool',
                        default=True,
                        help='Whether to use the PERSON based entity in the data.')

    parser.add_argument('--use_location',
                        type='bool',
                        default=True,
                        help='Whether to use the LOCATION based entity in the data.')

    parser.add_argument('--use_org',
                        type='bool',
                        default=True,
                        help='Whether to use the ORGANIZATION based entity in the data.')

    parser.add_argument('--use_misc',
                        type='bool',
                        default=True,
                        help='Whether to use the MISC based entity in the data.')

    parser.add_argument('--use_all',
                        type='bool',
                        default=True,
                        help='Use all entity types in the data.')

    parser.add_argument('--use_hl_once',
                        type='bool',
                        default=False,
                        help='Whether to use a given highlight only once, or all its permutations.')

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

    # INTERMEDIATE FILE NAMES, W/NER, AND MASK INFO

    parser.add_argument('--train',
                        type=str,
                        default="train.json",
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

    # MODEL READY FILES

    parser.add_argument('--train_model',
                        type=str,
                        default="train_model.json",
                        help='Training Data ready for input to the model')

    parser.add_argument('--test_model',
                        type=str,
                        default="test_model.json",
                        help='Test Data ready for input to the model')

    parser.add_argument('--dev_model',
                        type=str,
                        default="dev_model.json",
                        help='Dev Data ready for input to the model')

    return parser.parse_args()