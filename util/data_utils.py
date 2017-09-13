import gensim
import numpy as np


def load_weights(args):
    inp = open(args.emb_file_path, 'r')

    idx = 1
    words = {}
    inverse_dict = {}

    words['UNK'] = 0
    inverse_dict[0] = 'UNK'

    vocab = open(args.vocab_file, 'r')

    for line in vocab:
        words[line.strip()] = idx
        inverse_dict[idx] = line.strip()
        idx += 1

    embeddings = np.zeros((args.vocab_size, args.embedding_dim), dtype='float32')

    for line in inp:
        tokens = line.split(' ')
        if tokens[0] in words:
            for i in range(0, len(tokens)-1):
                embeddings[words[tokens[0]], i] = float(tokens[i+1])

    return words, embeddings, inverse_dict


def create_w2v_model(args, data_ls):
    model = gensim.models.Word2Vec(data_ls, size=args.embedding_dim, min_count=5)  # Default
    model.wv.save_word2vec_format(fname=args.word_model, binary=True)

    return model


def init_dict(w2v_model):
    idx = 1
    new_dict = {}

    for key in w2v_model.vocab.keys():
        new_dict[key] = idx
        idx += 1

    return new_dict
