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
                        default='../data/emb/glove.6B.100d.txt',
                        help='glove model name and path')

    parser.add_argument('--trained_emb',
                        type=str,
                        default='../data/emb/trained/',
                        help='Path for pretrained word embs')

    parser.add_argument('--stopwords',
                        type=str,
                        default='../data/stopwords.txt',
                        help='List of stopwords')

    parser.add_argument('--source',
                        type=str,
                        default='cnn',
                        help='Data source cnn/dm')

    parser.add_argument('--pretrain',
                        type='bool',
                        default=False,
                        help='Pretrain Generator')

    parser.add_argument('--bilinear',
                        type='bool',
                        default=True,
                        help='Bilinear attn.')

    parser.add_argument('--bigram_m',
                        type='bool',
                        default=False,
                        help='Toggle bigram masks and parse tree masks')

    parser.add_argument('--is_root',
                        type='bool',
                        default=False,
                        help='Entities are root words only')

    parser.add_argument('--bigram_toggle',
                        type='bool',
                        default=False,
                        help='')

    parser.add_argument('--batch_data',
                        type='bool',
                        default=False,
                        help='Pre_batch')

    parser.add_argument('--probs_only',
                        type='bool',
                        default=False,
                        help='Look at summaries based on non sampled data')

    parser.add_argument('--full_test',
                        type='bool',
                        default=True,
                        help='Process full selection of CNN data')

    parser.add_argument('--bigram_loss',
                        type='bool',
                        default=False,
                        help='Include Bigram Loss inside objective')

    parser.add_argument("--sanity_check",
                        type='bool',
                        default=False,
                        help="Test overfitting on a single/few samples"
                        )

    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Size of word vectors')

    parser.add_argument('--pt_pad_id',
                        type=int,
                        default=236,
                        help='<padding> for parse tree labels')

    parser.add_argument('--vocab_size',
                        type=int,
                        default=150000,
                        help='Vocab size')

    parser.add_argument('--inp_len',
                        type=int,
                        default=100,
                        help='length of single sentence')

    parser.add_argument('--hl_len',
                        type=int,
                        default=25,
                        help='length of single sentence in highlights')

    parser.add_argument('--pt_len',
                        type=int,
                        default=15,
                        help='length of single sentence in highlights')

    parser.add_argument('--nclasses',
                        type=int,
                        default=2481,
                        help='Number of unique entities')

    parser.add_argument("--train_output_readable",
                        type=str,
                        default="../data/results/readable/",
                        help="path to annotated train results"
                        )

    parser.add_argument("--train_output_mask",
                        type=str,
                        default="../data/results/masks/",
                        help="path to annotated train results"
                        )

    parser.add_argument("--system_summ_path",
                        type=str,
                        default="../data/results/summaries/system/",
                        help="system summaries"
                        )

    parser.add_argument("--model_summ_path",
                        type=str,
                        default="../data/results/summaries/model/",
                        help="gold standard summaries"
                        )

    parser.add_argument('--pad_repeat',
                        type='bool',
                        default=False,
                        help='Adhere to strict input Q len, by padding using repeated values.')

    parser.add_argument("--rouge_dir",
                        type=str,
                        default="../data/results/summaries/rouge/",
                        help="Rouge Outputs"
                        )

    parser.add_argument("--rouge_tmp",
                        type=str,
                        default="../data/results/tmp/",
                        help="Rouge Tmp"
                        )

    parser.add_argument("--save_model",
                        type=str,
                        default="save_models/",
                        help="path to save model parameters"
                        )

    parser.add_argument("--weight_eval",
                        type=str,
                        default="../data/results/weights/",
                        help="path to save model parameters, for encoder weights"
                        )

    parser.add_argument("--batch_dir",
                        type=str,
                        default="../data/batches/",
                        help="Directory to keep batched data on HDD"
                        )

    parser.add_argument("--online_batch_size",
                        type=int,
                        default=10,
                        help="Number of batches to have loaded onto ram at a given time"
                        )

    parser.add_argument("--num_files_train",
                        type=int,
                        default=12,
                        help="Number of batches to have loaded onto ram at a given time"
                        )

    parser.add_argument("--num_files_dev",
                        type=int,
                        default=1,
                        help="Number of batches to have loaded onto ram at a given time - dev"
                        )

    parser.add_argument("--num_files_test",
                        type=int,
                        default=1,
                        help="Number of batches to have loaded onto ram at a given time - dev"
                        )

    parser.add_argument("--load_model",
                        type=str,
                        default="",
                        help="path to load model"
                        )

    parser.add_argument("--load_model_pretrain",
                        type='bool',
                        default=False,
                        help="Train full model with pre-trained Generator"
                        )

    parser.add_argument("--train",
                        type=str,
                        default="../data/training_model.json",
                        help="path to training data"
                        )

    parser.add_argument("--entities",
                        type=str,
                        default="entities.json",
                        help="path to entity data"
                        )

    parser.add_argument("--dev",
                        type=str,
                        default="../data/dev_model.json",
                        help="path to development data"
                        )

    parser.add_argument("--dev_baseline",
                        type='bool',
                        default=False,
                        help="Get baseline ROUGE for dev set, based on BG."
                        )

    parser.add_argument("--test_baseline",
                        type='bool',
                        default=False,
                        help="Get baseline ROUGE for test set, based on BG."
                        )

    parser.add_argument("--test",
                        type=str,
                        default="../data/test_model.json",
                        help="path to test data"
                        )

    parser.add_argument("--max_epochs",
                        type=int,
                        default=25,
                        help="maximum # of epochs"
                        )

    parser.add_argument("--batch",
                        type=int,
                        default=256,
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
                        default=0.2,
                        help="dropout rate"
                        )

    parser.add_argument("--l2_reg",
                        type=float,
                        default=1e-6,
                        help="L2 regularization weight"
                        )

    parser.add_argument("-act", "--activation",
                        type=str,
                        default="tanh",
                        help="type of activation function"
                        )

    parser.add_argument("-d", "--hidden_dimension",
                        type=int,
                        default=128,
                        help="hidden dimension"
                        )

    parser.add_argument("--n",
                        type=int,
                        default=4,
                        help="HL per sample"
                        )

    parser.add_argument("--bigram_smoothing",
                        type=float,
                        default=1e-8,
                        help="Prevent div by 0"
                        )

    parser.add_argument("-d2", "--hidden_dimension2",
                        type=int,
                        default=30,
                        help="hidden dimension"
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

    parser.add_argument("--layer",
                           type=str,
                           default="lstm",
                           help="type of recurrent layer"
                           )

    parser.add_argument("--use_all",
                        type=int,
                        default=1,
                        help="whether to use the states of all layers"
                        )

    parser.add_argument("--coeff_z",
                        type=float,
                        default=10.0,
                        help="Scaling in encoder loss"
                        )

    parser.add_argument("--coeff_adequacy",
                        type=float,
                        default=10.0,
                        help="Scaling in encoder loss"
                        )

    parser.add_argument("--coeff_cost_scale",
                        type=float,
                        default=1.0
                        )

    parser.add_argument("--z_perc",
                        type=float,
                        default=0.4,
                        help="z_perc"
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

    parser.add_argument("--sparsity",
                           type=float,
                           default=0.0003
                           )
    parser.add_argument("--coherent",
                           type=float,
                           default=2.0
                           )

    return parser.parse_args()
