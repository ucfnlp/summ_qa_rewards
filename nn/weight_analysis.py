import gzip
import pickle
import os.path
import numpy as np

import matplotlib.pyplot as plt

import summarization_args


def weight_analysis(args):
    norms = []
    avg = []
    var = []
    weights = []
    early_term = False

    for e in xrange(args.max_epochs):
        for i in xrange(args.batch):
            path = args.weight_eval + 'e_' + str(e) + '_b_' + str(i) + '_weights.pkl.gz'

            if not os.path.isfile(path):
                early_term = True
                break

            f = gzip.open(path, 'rb')

            a = pickle.load(f)
            values = []
            for item in a:
                values.extend(item.ravel().tolist())

            weights.append(values)
            norms.append(np.linalg.norm(values, ord=1))
            avg.append(np.mean(values))
            var.append(np.var(values))

            f.close()

        plt.hist(weights[len(weights)-1], bins=20)
        plt.xlabel('W')
        plt.savefig('../data/results/plots/e_' + str(e+1) + '.png')
        plt.clf()
        plt.close()

        if early_term:
            break

    x = range(0,len(norms))
    plt.ylabel('L1 Norm')
    plt.xlabel('Num Batches')
    plt.plot(x, norms)
    plt.savefig('../data/results/plots/norms.png')

    plt.clf()
    plt.close()

    plt.ylabel('W Average')
    plt.xlabel('Num Batches')
    plt.plot(x, avg)
    plt.savefig('../data/results/plots/avg.png')

    plt.clf()
    plt.close()

    plt.ylabel('W Variance')
    plt.xlabel('Num Batches')
    plt.plot(x, var)
    plt.savefig('../data/results/plots/variance.png')

    plt.clf()
    plt.close()


if __name__ == "__main__":
    args = summarization_args.get_args()
    weight_analysis(args)