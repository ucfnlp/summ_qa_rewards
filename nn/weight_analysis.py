import gzip
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import summarization_args


def weight_analysis(args):
    norms = []

    for e in xrange(args.max_epochs):
        for i in xrange(args.batch):
            f = gzip.open(args.weight_eval + 'e_' + str(e) + '_b_' + str(i) + '_weights.pkl.gz', 'rb')

            a = pickle.load(f)
            values = []
            for item in a:
                values.extend(item.ravel().tolist())
            norms.append(np.linalg.norm(values, ord=1))

            f.close()

    x = range(0,len(norms))
    plt.plot(x, norms)
    plt.show()


if __name__ == "__main__":
    args = summarization_args.get_args()
    weight_analysis(args)